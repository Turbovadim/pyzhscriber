use anyhow::{Context, Result, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Sample, SampleFormat, Stream, StreamConfig};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

const TARGET_SAMPLE_RATE: u32 = 16_000;
/// How often the worker moves captured audio out of the callback's buffer and
/// resamples it, so stopping only has the last fraction of a second left.
const DRAIN_INTERVAL: Duration = Duration::from_millis(50);

/// Receives 16 kHz audio while recording, a piece at a time. It is dropped
/// once the last sample has been passed on, which marks the end of input.
pub type AudioTap = Box<dyn FnMut(&[f32]) + Send>;

/// Meter levels shared lock-free between the audio callback and the UI.
#[derive(Default)]
struct Levels([AtomicU32; 9]);

impl Levels {
    fn load(&self) -> [f32; 9] {
        self.0
            .each_ref()
            .map(|level| f32::from_bits(level.load(Ordering::Relaxed)))
    }

    fn store(&self, levels: [f32; 9]) {
        for (slot, level) in self.0.iter().zip(levels) {
            slot.store(level.to_bits(), Ordering::Relaxed);
        }
    }
}

struct RecordingState {
    stream: Stream,
    /// Filled by the realtime callback. The worker swaps it with `spare`, so
    /// both keep their capacity and the callback does not reallocate.
    captured: Arc<Mutex<Vec<f32>>>,
    spare: Vec<f32>,
    converter: Converter,
    tap: Option<AudioTap>,
    /// Converted samples already handed to `tap`.
    tapped: usize,
}

impl RecordingState {
    fn drain(&mut self) {
        std::mem::swap(&mut *self.captured.lock(), &mut self.spare);
        self.converter.push(&self.spare);
        self.spare.clear();
        if let Some(tap) = &mut self.tap {
            let fresh = &self.converter.output[self.tapped..];
            if !fresh.is_empty() {
                tap(fresh);
                self.tapped = self.converter.output.len();
            }
        }
    }

    fn finish(mut self) -> Result<Vec<f32>> {
        drop(self.stream);
        std::mem::swap(&mut *self.captured.lock(), &mut self.spare);
        self.converter.push(&self.spare);
        let audio = self.converter.finish()?;
        if let (Some(mut tap), Some(rest)) = (self.tap, audio.get(self.tapped..)) {
            tap(rest);
        }
        Ok(audio)
    }
}

/// Turns native-rate mono into 16 kHz audio while recording is in progress.
struct Converter {
    input: Vec<f32>,
    output: Vec<f32>,
    resampler: Option<Resampler>,
}

impl Converter {
    fn new(input_rate: u32) -> Self {
        Self {
            input: Vec::new(),
            output: Vec::with_capacity(TARGET_SAMPLE_RATE as usize * 30),
            resampler: Resampler::new(input_rate, TARGET_SAMPLE_RATE),
        }
    }

    fn push(&mut self, samples: &[f32]) {
        match &mut self.resampler {
            Some(resampler) => {
                self.input.extend_from_slice(samples);
                resampler.advance(&self.input, &mut self.output, false);
            }
            None => self.output.extend_from_slice(samples),
        }
    }

    fn finish(mut self) -> Result<Vec<f32>> {
        if self.input.is_empty() && self.output.is_empty() {
            return Err(anyhow!("the microphone returned no audio"));
        }
        match &mut self.resampler {
            Some(_) if self.input.len() < 2 => Ok(self.input),
            Some(resampler) => {
                resampler.advance(&self.input, &mut self.output, true);
                Ok(self.output)
            }
            None => Ok(self.output),
        }
    }
}

pub struct AudioRecorder {
    commands: mpsc::Sender<AudioCommand>,
    recording: Arc<AtomicBool>,
    levels: Arc<Levels>,
}

enum AudioCommand {
    Start {
        selected_device: Option<String>,
        tap: Option<AudioTap>,
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
        let levels = Arc::new(Levels::default());
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

    pub fn start(&self, selected_device: Option<&str>, tap: Option<AudioTap>) -> Result<()> {
        if self.is_recording() {
            return Ok(());
        }
        let (reply, result) = mpsc::channel();
        self.commands
            .send(AudioCommand::Start {
                selected_device: selected_device.map(str::to_string),
                tap,
                reply,
            })
            .map_err(|_| anyhow!("audio worker stopped"))?;
        result.recv().context("audio worker stopped")?
    }

    pub fn stop(&self) -> Result<Vec<f32>> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(AudioCommand::Stop { reply })
            .map_err(|_| anyhow!("audio worker stopped"))?;
        result.recv().context("audio worker stopped")?
    }

    pub fn cancel(&self) {
        let _ = self.commands.send(AudioCommand::Cancel);
    }

    pub fn is_recording(&self) -> bool {
        self.recording.load(Ordering::SeqCst)
    }

    pub fn levels(&self) -> [f32; 9] {
        self.levels.load()
    }
}

fn audio_worker(
    receiver: mpsc::Receiver<AudioCommand>,
    recording: Arc<AtomicBool>,
    levels: Arc<Levels>,
) {
    let host = cpal::default_host();
    let mut named_device: Option<(String, Device)> = None;
    let mut state: Option<RecordingState> = None;
    loop {
        let command = match state.as_mut() {
            Some(active) => match receiver.recv_timeout(DRAIN_INTERVAL) {
                Ok(command) => command,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    active.drain();
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match receiver.recv() {
                Ok(command) => command,
                Err(_) => break,
            },
        };
        match command {
            AudioCommand::Start {
                selected_device,
                tap,
                reply,
            } => {
                let result = open_recording(
                    &host,
                    selected_device.as_deref(),
                    &mut named_device,
                    &levels,
                )
                .map(|mut stream| {
                    stream.tap = tap;
                    state = Some(stream);
                    recording.store(true, Ordering::SeqCst);
                });
                let _ = reply.send(result);
            }
            AudioCommand::Stop { reply } => {
                let result = state
                    .take()
                    .context("recording is not active")
                    .and_then(RecordingState::finish);
                recording.store(false, Ordering::SeqCst);
                levels.store([0.0; 9]);
                let _ = reply.send(result);
            }
            AudioCommand::Cancel => {
                state.take();
                recording.store(false, Ordering::SeqCst);
                levels.store([0.0; 9]);
            }
        }
    }
}

/// Opens the selected microphone, reusing the device found for that name last
/// time so a press does not enumerate every input. A cached device that fails
/// to open is looked up again once, in case it was unplugged and reconnected.
fn open_recording(
    host: &Host,
    selected: Option<&str>,
    named_device: &mut Option<(String, Device)>,
    levels: &Arc<Levels>,
) -> Result<RecordingState> {
    let cached = selected.and_then(|name| {
        named_device
            .as_ref()
            .filter(|(cached, _)| cached == name)
            .map(|(_, device)| device.clone())
    });
    if let Some(device) = cached {
        match start_stream(&device, levels.clone()) {
            Ok(state) => return Ok(state),
            Err(error) => {
                log::warn!("cached microphone failed to open, looking it up again: {error:#}");
                *named_device = None;
            }
        }
    }
    let device = match selected {
        Some(name) => {
            let found = host
                .input_devices()?
                .find(|device| device.name().as_deref() == Ok(name));
            if let Some(device) = &found {
                *named_device = Some((name.to_string(), device.clone()));
            }
            found.or_else(|| host.default_input_device())
        }
        None => host.default_input_device(),
    }
    .context("no microphone is available")?;
    start_stream(&device, levels.clone())
}

fn start_stream(device: &Device, levels: Arc<Levels>) -> Result<RecordingState> {
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate.0;
    let channels = usize::from(config.channels);
    // A second per buffer is far more than one drain interval, so the callback
    // only reallocates if the worker stalls for that long.
    let capacity = sample_rate as usize;
    let captured = Arc::new(Mutex::new(Vec::with_capacity(capacity)));
    let error_callback = |error| log::error!("microphone stream error: {error}");

    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(
            device,
            &config,
            channels,
            captured.clone(),
            levels,
            error_callback,
        )?,
        SampleFormat::I16 => build_stream::<i16>(
            device,
            &config,
            channels,
            captured.clone(),
            levels,
            error_callback,
        )?,
        SampleFormat::U16 => build_stream::<u16>(
            device,
            &config,
            channels,
            captured.clone(),
            levels,
            error_callback,
        )?,
        format => return Err(anyhow!("unsupported microphone sample format: {format:?}")),
    };
    stream.play()?;
    Ok(RecordingState {
        stream,
        captured,
        spare: Vec::with_capacity(capacity),
        converter: Converter::new(sample_rate),
        tap: None,
        tapped: 0,
    })
}

fn build_stream<T>(
    device: &Device,
    config: &StreamConfig,
    channels: usize,
    captured: Arc<Mutex<Vec<f32>>>,
    levels: Arc<Levels>,
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
    let mut smoothed = [0.0_f32; 9];
    let stream = device.build_input_stream(
        config,
        move |input: &[T], _| {
            let mut output = captured.lock();
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
            for index in 0..9 {
                let rms = if counts[index] == 0 {
                    0.0
                } else {
                    (energy[index] / counts[index] as f32).sqrt()
                };
                let target = ((rms - gate) / range).clamp(0.0, 1.0);
                smoothed[index] = smoothed[index] * 0.62 + target * 0.38;
            }
            levels.store(smoothed);
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
    let Some(mut resampler) = Resampler::new(input_rate, output_rate) else {
        return input.to_vec();
    };
    if input.len() < 2 {
        return input.to_vec();
    }
    let mut output = Vec::with_capacity((input.len() as f64 * resampler.ratio) as usize);
    resampler.advance(input, &mut output, true);
    output
}

/// One tabulated phase of the windowed sinc.
struct Kernel {
    taps: Vec<f32>,
    /// The taps' total, which normalises a sample whose kernel lies wholly
    /// inside the input.
    weight: f32,
}

/// Incremental form of [`resample`]: input can arrive in pieces and each output
/// sample is produced as soon as all of its taps are available.
pub(crate) struct Resampler {
    ratio: f64,
    reach: usize,
    kernels: Vec<Kernel>,
    produced: usize,
}

impl Resampler {
    /// `None` when the rates match and no conversion is needed.
    pub(crate) fn new(input_rate: u32, output_rate: u32) -> Option<Self> {
        if input_rate == output_rate {
            return None;
        }
        let ratio = output_rate as f64 / input_rate as f64;
        // Cycles per input sample, held below the lower of the two Nyquist
        // limits so the filter has room to roll off before reaching it.
        let cutoff = 0.45 * ratio.min(1.0);
        let half_width = RESAMPLE_ZERO_CROSSINGS / (2.0 * cutoff);
        let reach = half_width as usize;
        Some(Self {
            ratio,
            reach,
            kernels: phase_kernels(cutoff, half_width, reach),
            produced: 0,
        })
    }

    /// Appends the output samples that `input` (everything received so far)
    /// fully covers. With `finished`, also the tail, whose kernels hang off the
    /// end of the buffer.
    pub(crate) fn advance(&mut self, input: &[f32], output: &mut Vec<f32>, finished: bool) {
        let end = if finished {
            (input.len() as f64 * self.ratio) as usize
        } else {
            usize::MAX
        };
        while self.produced < end {
            let (base, phase) = self.tap_position(self.produced);
            if !finished && base + self.reach as isize >= input.len() as isize {
                break;
            }
            output.push(self.sample(input, base, phase));
            self.produced += 1;
        }
    }

    fn tap_position(&self, index: usize) -> (isize, usize) {
        let center = index as f64 / self.ratio;
        let floor = center.floor();
        let scaled = ((center - floor) * RESAMPLE_PHASES as f64).round() as usize;
        // Rounding up to a whole sample belongs to the next tap, not phase 0 of
        // this one.
        if scaled >= RESAMPLE_PHASES {
            (floor as isize + 1, 0)
        } else {
            (floor as isize, scaled)
        }
    }

    fn sample(&self, input: &[f32], base: isize, phase: usize) -> f32 {
        let kernel = &self.kernels[phase];
        let start = base - self.reach as isize;
        let window = usize::try_from(start)
            .ok()
            .and_then(|start| input.get(start..start + kernel.taps.len()));
        let (sum, weight) = match window {
            Some(window) => (dot(&kernel.taps, window), kernel.weight),
            None => {
                let mut sum = 0.0;
                let mut weight = 0.0;
                for (tap, coefficient) in kernel.taps.iter().enumerate() {
                    let position = start + tap as isize;
                    if position < 0 || position >= input.len() as isize {
                        continue;
                    }
                    sum += coefficient * input[position as usize];
                    weight += coefficient;
                }
                (sum, weight)
            }
        };
        // Normalising by the taps actually used keeps unity gain at the edges,
        // where part of the kernel hangs off the buffer.
        if weight.abs() > f32::EPSILON {
            sum / weight
        } else {
            0.0
        }
    }
}

/// Eight independent accumulators, so the compiler can vectorise the
/// multiply-accumulate instead of chaining every addition.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    let (left_chunks, left_rest) = left.as_chunks::<8>();
    let (right_chunks, right_rest) = right.as_chunks::<8>();
    let mut lanes = [0.0_f32; 8];
    for (left, right) in left_chunks.iter().zip(right_chunks) {
        for lane in 0..8 {
            lanes[lane] += left[lane] * right[lane];
        }
    }
    let rest: f32 = left_rest
        .iter()
        .zip(right_rest)
        .map(|(left, right)| left * right)
        .sum();
    lanes.iter().sum::<f32>() + rest
}

fn phase_kernels(cutoff: f64, half_width: f64, reach: usize) -> Vec<Kernel> {
    let reach = reach as isize;
    (0..RESAMPLE_PHASES)
        .map(|phase| {
            let fraction = phase as f64 / RESAMPLE_PHASES as f64;
            let taps: Vec<f32> = (0..=reach * 2)
                .map(|tap| {
                    // Distance from the point being interpolated, in input samples.
                    let offset = (tap - reach) as f64 - fraction;
                    let value =
                        2.0 * cutoff * sinc(2.0 * cutoff * offset) * blackman(offset / half_width);
                    value as f32
                })
                .collect();
            let weight = taps.iter().sum();
            Kernel { taps, weight }
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

/// Encodes 16 kHz mono samples as 16-bit PCM WAV.
pub fn write_wav(audio: &[f32], out: impl std::io::Write + std::io::Seek) -> Result<()> {
    let mut writer = hound::WavWriter::new(
        out,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in audio {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
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

    #[test]
    fn resampling_in_pieces_matches_one_pass() {
        let input = tone(440.0, 44_100, 20_000);
        let whole = resample(&input, 44_100, 16_000);

        let mut resampler = Resampler::new(44_100, 16_000).unwrap();
        let mut received = Vec::new();
        let mut pieces = Vec::new();
        for chunk in input.chunks(441) {
            received.extend_from_slice(chunk);
            resampler.advance(&received, &mut pieces, false);
        }
        resampler.advance(&received, &mut pieces, true);
        assert_eq!(pieces, whole);
    }
}
