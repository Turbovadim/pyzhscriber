use crate::audio::{AudioRecorder, resample};
use crate::models::ModelRegistry;
use crate::permissions::{self, PermissionStatuses};
use crate::settings::{SettingsStore, TranscriptionMode};
use crate::transcription::Transcriber;
use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "pyzhscriber", version, about)]
struct Arguments {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe a WAV file without opening the app window.
    Transcribe {
        file: PathBuf,
        #[arg(long, value_enum)]
        mode: Option<Mode>,
        /// Catalog model ID. Supplying one implies single-model mode.
        #[arg(long)]
        model: Option<String>,
    },
    /// Print macOS permission state and run a short microphone capture test.
    Doctor,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Single,
    Routed,
}

pub fn run_if_requested() -> Result<bool> {
    let arguments = Arguments::parse();
    let Some(command) = arguments.command else {
        return Ok(false);
    };
    match command {
        Command::Transcribe { file, mode, model } => transcribe(file, mode, model)?,
        Command::Doctor => doctor()?,
    }
    Ok(true)
}

fn doctor() -> Result<()> {
    let statuses = PermissionStatuses::current();
    println!("executable: {}", std::env::current_exe()?.display());
    println!("microphone: {}", statuses.microphone.label());
    println!("accessibility: {}", statuses.accessibility.label());
    println!(
        "input monitoring: {} (required for the shortcut to work in other apps)",
        statuses.input_monitoring.label()
    );
    println!(
        "keyboard event tap: {}",
        if permissions::event_tap_available() {
            "available"
        } else {
            "unavailable"
        }
    );
    println!("input devices: {:?}", AudioRecorder::input_devices());

    let settings = SettingsStore::new()?.load()?;
    let recorder = AudioRecorder::new();
    recorder.start(settings.selected_microphone.as_deref())?;
    std::thread::sleep(std::time::Duration::from_millis(1500));
    let audio = recorder.stop()?;
    let peak = audio
        .iter()
        .fold(0.0_f32, |peak, sample| peak.max(sample.abs()));
    println!(
        "microphone test: {} samples, peak {peak:.4}{}",
        audio.len(),
        if peak == 0.0 {
            " (digital silence: macOS is not delivering audio to this process)"
        } else {
            ""
        }
    );
    Ok(())
}

fn transcribe(file: PathBuf, mode: Option<Mode>, model: Option<String>) -> Result<()> {
    let mut settings = SettingsStore::new()?.load()?;
    if let Some(mode) = mode {
        settings.transcription_mode = match mode {
            Mode::Single => TranscriptionMode::Single,
            Mode::Routed => TranscriptionMode::Routed,
        };
    }
    if let Some(model) = model {
        settings.transcription_mode = TranscriptionMode::Single;
        settings.selected_model = model;
        settings.allowed_languages.clear();
    }
    let audio = read_wav(&file)?;
    let transcriber = Transcriber::new(Arc::new(ModelRegistry::new()?));
    let text = transcriber.transcribe(&audio, &settings)?;
    println!("{text}");
    Ok(())
}

fn read_wav(path: &PathBuf) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open WAV file {}", path.display()))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        return Err(anyhow!("WAV file has no channels"));
    }
    let interleaved = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, bits) if bits <= 16 => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| value as f32 / i16::MAX as f32))
            .collect::<Result<Vec<_>, _>>()?,
        (hound::SampleFormat::Float, 32) => {
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
        _ => {
            return Err(anyhow!(
                "unsupported WAV format: {:?}, {} bit",
                spec.sample_format,
                spec.bits_per_sample
            ));
        }
    };
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect::<Vec<_>>();
    Ok(resample(&mono, spec.sample_rate, 16_000))
}
