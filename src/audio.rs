use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

const TARGET_SAMPLE_RATE: u32 = 16_000;

struct RecordingState {
    stream: Option<Stream>,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

pub struct AudioRecorder {
    commands: mpsc::Sender<AudioCommand>,
    recording: Arc<AtomicBool>,
    levels: Arc<Mutex<[f32; 9]>>,
}

enum AudioCommand {
    Start {
        selected_device: Option<String>,
        reply: mpsc::Sender<Result<()>>,
    },
    Stop {
        reply: mpsc::Sender<Result<Vec<f32>>>,
    },
    Cancel,
}

impl AudioRecorder {
    pub fn new() -> Self {
        let (commands, receiver) = mpsc::channel();
        let recording = Arc::new(AtomicBool::new(false));
        let levels = Arc::new(Mutex::new([0.0; 9]));
        let recording_for_worker = recording.clone();
        let levels_for_worker = levels.clone();
        std::thread::Builder::new()
            .name("pyzhscriber-audio".into())
            .spawn(move || audio_worker(receiver, recording_for_worker, levels_for_worker))
            .expect("start audio worker");
        Self {
            commands,
            recording,
            levels,
        }
    }

    pub fn input_devices() -> Vec<String> {
        cpal::default_host()
            .input_devices()
            .map(|devices| devices.filter_map(|device| device.name().ok()).collect())
            .unwrap_or_default()
    }

    pub fn start(&self, selected_device: Option<&str>) -> Result<()> {
        if self.is_recording() {
            return Ok(());
        }
        let (reply, result) = mpsc::channel();
        self.commands.send(AudioCommand::Start {
            selected_device: selected_device.map(str::to_string),
            reply,
        })?;
        result.recv().context("audio worker stopped")?
    }

    pub fn stop(&self) -> Result<Vec<f32>> {
        let (reply, result) = mpsc::channel();
        self.commands.send(AudioCommand::Stop { reply })?;
        result.recv().context("audio worker stopped")?
    }

    pub fn cancel(&self) {
        let _ = self.commands.send(AudioCommand::Cancel);
    }

    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }

    pub fn levels(&self) -> [f32; 9] {
        *self.levels.lock()
    }
}

fn audio_worker(
    receiver: mpsc::Receiver<AudioCommand>,
    recording: Arc<AtomicBool>,
    levels: Arc<Mutex<[f32; 9]>>,
) {
    let mut state: Option<RecordingState> = None;
    while let Ok(command) = receiver.recv() {
        match command {
            AudioCommand::Start {
                selected_device,
                reply,
            } => {
                let result =
                    start_stream(selected_device.as_deref(), levels.clone()).map(|stream| {
                        state = Some(stream);
                        recording.store(true, Ordering::SeqCst);
                    });
                let _ = reply.send(result);
            }
            AudioCommand::Stop { reply } => {
                let result = stop_stream(state.take());
                recording.store(false, Ordering::SeqCst);
                *levels.lock() = [0.0; 9];
                let _ = reply.send(result);
            }
            AudioCommand::Cancel => {
                state.take();
                recording.store(false, Ordering::SeqCst);
                *levels.lock() = [0.0; 9];
            }
        }
    }
}

fn start_stream(
    selected_device: Option<&str>,
    levels: Arc<Mutex<[f32; 9]>>,
) -> Result<RecordingState> {
    let host = cpal::default_host();
    let device = match selected_device {
        Some(name) => host
            .input_devices()?
            .find(|device| device.name().as_deref() == Ok(name))
            .or_else(|| host.default_input_device()),
        None => host.default_input_device(),
    }
    .context("no microphone is available")?;
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate.0;
    let channels = usize::from(config.channels);
    let samples = Arc::new(Mutex::new(Vec::with_capacity(sample_rate as usize * 30)));
    let error_callback = |error| log::error!("microphone stream error: {error}");

    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(
            &device,
            &config,
            channels,
            samples.clone(),
            levels,
            error_callback,
        )?,
        SampleFormat::I16 => build_stream::<i16>(
            &device,
            &config,
            channels,
            samples.clone(),
            levels,
            error_callback,
        )?,
        SampleFormat::U16 => build_stream::<u16>(
            &device,
            &config,
            channels,
            samples.clone(),
            levels,
            error_callback,
        )?,
        format => return Err(anyhow!("unsupported microphone sample format: {format:?}")),
    };
    stream.play()?;
    Ok(RecordingState {
        stream: Some(stream),
        samples,
        sample_rate,
    })
}

fn stop_stream(state: Option<RecordingState>) -> Result<Vec<f32>> {
    let mut recording = state.context("recording is not active")?;
    recording.stream.take();
    let samples = recording.samples.lock().clone();
    if samples.is_empty() {
        return Err(anyhow!("the microphone returned no audio"));
    }
    Ok(resample(
        &samples,
        recording.sample_rate,
        TARGET_SAMPLE_RATE,
    ))
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: usize,
    samples: Arc<Mutex<Vec<f32>>>,
    levels: Arc<Mutex<[f32; 9]>>,
    error_callback: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<Stream>
where
    T: cpal::SizedSample + Copy + Send + 'static,
    f32: cpal::FromSample<T>,
{
    // Running estimate of the room's noise floor. It falls quickly to any quiet
    // moment and climbs slowly through speech, so speech never becomes the
    // floor while room tone and mic self-noise do.
    let mut noise = 0.0_f32;
    let stream = device.build_input_stream(
        config,
        move |input: &[T], _| {
            let mut output = samples.lock();
            let frame_count = input.len() / channels;
            let mut energy = [0.0_f32; 9];
            let mut counts = [0_usize; 9];
            for (index, frame) in input.chunks(channels).enumerate() {
                let sum: f32 = frame.iter().copied().map(f32::from_sample).sum();
                let mono = sum / channels as f32;
                output.push(mono);
                let bucket = (index * 9 / frame_count.max(1)).min(8);
                energy[bucket] += mono * mono;
                counts[bucket] += 1;
            }
            drop(output);
            let frame_rms = (energy.iter().sum::<f32>() / frame_count.max(1) as f32).sqrt();
            let rate = if frame_rms < noise { 0.05 } else { 0.005 };
            noise += (frame_rms - noise) * rate;
            // The meter starts a safe margin above the floor, never below
            // -48 dBFS, and reaches full scale at -18 dBFS.
            let gate = (noise * 2.0).max(0.004);
            let range = (0.125 - gate).max(0.02);
            let mut current = levels.lock();
            for index in 0..9 {
                let rms = if counts[index] == 0 {
                    0.0
                } else {
                    (energy[index] / counts[index] as f32).sqrt()
                };
                let target = ((rms - gate) / range).clamp(0.0, 1.0);
                current[index] = current[index] * 0.62 + target * 0.38;
            }
        },
        error_callback,
        None,
    )?;
    Ok(stream)
}

/// Lifts quiet recordings toward the level transcription models expect. The gain
/// is capped so a nearly silent room is not amplified into noise, and the peak
/// stays below full scale to avoid clipping.
pub fn normalize_peak(samples: &mut [f32]) -> f32 {
    const TARGET_PEAK: f32 = 0.35;
    const MAX_GAIN: f32 = 12.0;

    let peak = samples
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    if peak <= 0.0 || peak >= TARGET_PEAK {
        return peak;
    }
    let gain = (TARGET_PEAK / peak).min(MAX_GAIN);
    for sample in samples.iter_mut() {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
    peak
}

/// Zero crossings of the sinc kept either side of each output sample. This sets
/// how sharply the filter rolls off, independently of the rate ratio.
const RESAMPLE_ZERO_CROSSINGS: f64 = 16.0;
/// Fractional sample offsets the kernel is tabulated at. Standard microphone
/// rates land on an exact phase; anything else is off by at most half of
/// 1/512th of a sample, which is far below anything audible.
const RESAMPLE_PHASES: usize = 256;

/// Band-limited resampling to the rate the transcription models expect.
///
/// Rate conversion has to remove everything above the new Nyquist first, or it
/// folds back into the speech band. At the usual 48 kHz microphone rate the
/// ratio to 16 kHz is exactly 3, so a plain interpolating resampler degenerates
/// into taking every third sample: a 12 kHz sibilant then reappears at full
/// strength on top of speech at 4 kHz. The windowed sinc below is the
/// anti-alias filter and the interpolator at once, tabulated per fractional
/// phase so the inner loop is only multiply-accumulate.
pub(crate) fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input_rate == output_rate || input.len() < 2 {
        return input.to_vec();
    }
    let ratio = output_rate as f64 / input_rate as f64;
    // Cycles per input sample, held below the lower of the two Nyquist limits
    // so the filter has room to roll off before reaching it.
    let cutoff = 0.45 * ratio.min(1.0);
    let half_width = RESAMPLE_ZERO_CROSSINGS / (2.0 * cutoff);
    let reach = half_width as isize;
    let kernels = phase_kernels(cutoff, half_width, reach);

    let output_len = (input.len() as f64 * ratio) as usize;
    (0..output_len)
        .map(|index| {
            let center = index as f64 / ratio;
            let floor = center.floor();
            let scaled = ((center - floor) * RESAMPLE_PHASES as f64).round() as usize;
            // Rounding up to a whole sample belongs to the next tap, not phase 0
            // of this one.
            let (base, phase) = if scaled >= RESAMPLE_PHASES {
                (floor as isize + 1, 0)
            } else {
                (floor as isize, scaled)
            };

            let mut sum = 0.0;
            let mut weight = 0.0;
            for (tap, coefficient) in kernels[phase].iter().enumerate() {
                let position = base - reach + tap as isize;
                if position < 0 || position >= input.len() as isize {
                    continue;
                }
                sum += coefficient * input[position as usize];
                weight += coefficient;
            }
            // Normalising by the taps actually used keeps unity gain at the
            // edges, where part of the kernel hangs off the buffer.
            if weight.abs() > f32::EPSILON {
                sum / weight
            } else {
                0.0
            }
        })
        .collect()
}

fn phase_kernels(cutoff: f64, half_width: f64, reach: isize) -> Vec<Vec<f32>> {
    (0..RESAMPLE_PHASES)
        .map(|phase| {
            let fraction = phase as f64 / RESAMPLE_PHASES as f64;
            (0..=reach * 2)
                .map(|tap| {
                    // Distance from the point being interpolated, in input samples.
                    let offset = (tap - reach) as f64 - fraction;
                    let value =
                        2.0 * cutoff * sinc(2.0 * cutoff * offset) * blackman(offset / half_width);
                    value as f32
                })
                .collect()
        })
        .collect()
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let scaled = std::f64::consts::PI * x;
        scaled.sin() / scaled
    }
}

/// Blackman window, which buys about 74 dB of stopband rejection.
fn blackman(position: f64) -> f64 {
    if position.abs() >= 1.0 {
        return 0.0;
    }
    let angle = std::f64::consts::PI * position;
    0.42 + 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizing_lifts_quiet_audio_without_clipping() {
        let mut quiet = vec![0.02, -0.02, 0.01];
        normalize_peak(&mut quiet);
        assert!(quiet.iter().all(|sample| sample.abs() <= 1.0));
        assert!(quiet[0] > 0.2);

        let mut near_silence = vec![0.0001, -0.0001];
        normalize_peak(&mut near_silence);
        assert!(near_silence[0] <= 0.0001 * 12.0 + f32::EPSILON);
    }

    fn tone(frequency: f64, rate: u32, samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|index| {
                (2.0 * std::f64::consts::PI * frequency * index as f64 / rate as f64).sin() as f32
            })
            .collect()
    }

    /// Ignores the ends, where part of the kernel hangs off the buffer.
    fn steady_peak(samples: &[f32]) -> f32 {
        samples[400..samples.len() - 400]
            .iter()
            .fold(0.0_f32, |peak, sample| peak.max(sample.abs()))
    }

    #[test]
    fn resampling_preserves_duration_and_the_speech_band() {
        let result = resample(&tone(440.0, 48_000, 48_000), 48_000, 16_000);
        assert_eq!(result.len(), 16_000);
        assert!((steady_peak(&result) - 1.0).abs() < 0.02);
    }

    #[test]
    fn resampling_rejects_content_above_the_new_nyquist() {
        // Decimating 48 kHz by three without filtering folds 12 kHz onto 4 kHz,
        // landing a sibilant in the middle of the speech band at full strength.
        let result = resample(&tone(12_000.0, 48_000, 48_000), 48_000, 16_000);
        let peak = steady_peak(&result);
        assert!(peak < 0.01, "12 kHz aliased through at {peak}");
    }
}
