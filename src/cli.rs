use crate::audio::{AudioRecorder, resample};
use crate::ipc::{self, Request, Response};
use crate::models::{ModelRegistry, display_size};
use crate::permissions::{self, PermissionStatuses};
use crate::settings::{AppSettings, SettingsStore, TranscriptionMode};
use crate::transcription::Transcriber;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rodio::Source;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "pyzhscriber", version, about)]
struct Arguments {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Transcribe audio through the running app (WAV, MP3, FLAC, Ogg/Vorbis/Opus, AAC, M4A/MP4).
    Transcribe {
        file: PathBuf,
        #[command(flatten)]
        options: TranscribeOptions,
        /// Load a separate model without requiring the running app.
        #[arg(long)]
        standalone: bool,
        /// Print a JSON response, including errors, to stdout.
        #[arg(long)]
        json: bool,
    },
    /// Check the running app and its loaded model.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Print macOS permission state and run a short microphone capture test.
    Doctor,
    /// Install the command for this user and configure zsh/bash PATH.
    InstallCli,
}

#[derive(Clone, Copy, Debug, ValueEnum, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Single,
    Routed,
}

#[derive(clap::Args, Debug, Default, Deserialize, Serialize)]
pub struct TranscribeOptions {
    #[arg(long, value_enum)]
    mode: Option<Mode>,
    /// Catalog model ID. Implies single-model mode and clears language limits.
    #[arg(long)]
    model: Option<String>,
}

impl TranscribeOptions {
    /// Choosing a local mode or model also turns cloud dictation off.
    pub fn apply(&self, settings: &mut AppSettings) {
        if self.mode.is_some() || self.model.is_some() {
            settings.cloud_enabled = false;
        }
        if let Some(mode) = self.mode {
            settings.transcription_mode = match mode {
                Mode::Single => TranscriptionMode::Single,
                Mode::Routed => TranscriptionMode::Routed,
            };
        }
        if let Some(model) = &self.model {
            settings.transcription_mode = TranscriptionMode::Single;
            settings.selected_model = model.clone();
            settings.allowed_languages.clear();
        }
    }
}

pub fn run_if_requested() -> Result<bool> {
    let arguments = Arguments::parse();
    let Some(command) = arguments.command else {
        return Ok(false);
    };
    match command {
        Command::Transcribe {
            file,
            options,
            standalone,
            json,
        } => {
            let result = if standalone {
                transcribe(file, &options).map(|text| Response::Transcript { text })
            } else {
                file.canonicalize()
                    .context("resolve input audio file")
                    .and_then(|file| ipc::request(&Request::Transcribe { file, options }))
            };
            print_response(result, json)?;
        }
        Command::Status { json } => print_response(ipc::request(&Request::Status), json)?,
        Command::Doctor => doctor()?,
        Command::InstallCli => println!("{}", crate::cli_install::install()?),
    }
    Ok(true)
}

fn print_response(result: Result<Response>, json: bool) -> Result<()> {
    let response = result.unwrap_or_else(|error| Response::Error {
        error: format!("{error:#}"),
    });
    if json {
        println!("{}", serde_json::to_string(&response)?);
    } else {
        match &response {
            Response::Transcript { text } => println!("{text}"),
            Response::Status {
                version,
                status,
                loaded_model,
                app_memory,
                model_weights,
            } => {
                println!(
                    "Pyzhscriber {version}: {status}\nLoaded model: {}",
                    loaded_model.as_deref().unwrap_or("none")
                );
                let parts: Vec<String> = app_memory
                    .map(|bytes| format!("{} app", display_size(bytes)))
                    .into_iter()
                    .chain(model_weights.map(|bytes| {
                        format!(
                            "{} model weights (mapped, in Cached Files)",
                            display_size(bytes)
                        )
                    }))
                    .collect();
                if !parts.is_empty() {
                    println!("Memory: {}", parts.join(" + "));
                }
            }
            Response::Error { .. } => {}
        }
    }
    if let Response::Error { error } = response {
        anyhow::bail!("{error}");
    }
    Ok(())
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
    recorder.start(settings.selected_microphone.as_deref(), None)?;
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

fn transcribe(file: PathBuf, options: &TranscribeOptions) -> Result<String> {
    let mut settings = SettingsStore::new()?.load()?;
    options.apply(&mut settings);
    let audio = read_audio(&file)?;
    let transcriber = Transcriber::new(Arc::new(ModelRegistry::new()?));
    let result = transcriber.transcribe(&audio, &settings);
    transcriber.shutdown();
    result
}

pub(crate) fn read_audio(path: &Path) -> Result<Vec<f32>> {
    anyhow::ensure!(path.is_file(), "input must be a regular audio file");
    let mut file =
        std::fs::File::open(path).with_context(|| format!("open audio file {}", path.display()))?;
    if let Some(audio) = crate::opus_audio::try_decode(&mut file)
        .with_context(|| format!("decode Opus file {}", path.display()))?
    {
        return Ok(audio);
    }
    // Probe contents rather than trusting the extension.
    let decoder = rodio::Decoder::try_from(file)
        .with_context(|| format!("decode audio file {}; supported formats: WAV, MP3, FLAC, Ogg/Vorbis/Opus, AAC, M4A/MP4 with AAC audio", path.display()))?;
    let sample_rate = decoder.sample_rate();
    let channels = usize::from(decoder.channels());
    anyhow::ensure!(
        sample_rate > 0 && channels > 0,
        "invalid audio sample rate or channel count"
    );
    let interleaved = decoder.collect::<Vec<f32>>();
    anyhow::ensure!(!interleaved.is_empty(), "file contains no decodable audio");
    anyhow::ensure!(
        interleaved.iter().all(|sample| sample.is_finite()),
        "audio contains non-finite samples"
    );
    let mono = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect::<Vec<_>>();
    Ok(resample(&mono, sample_rate, 16_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires ffmpeg to generate codec fixtures"]
    fn common_audio_formats_decode_and_resample() {
        let directory = std::env::temp_dir().join(format!("pyzh-formats-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        for (extension, codec) in [
            ("wav", "pcm_f32le"),
            ("mp3", "libmp3lame"),
            ("flac", "flac"),
            ("ogg", "vorbis"),
            ("opus", "libopus"),
            ("aac", "aac"),
            ("m4a", "aac"),
            ("mp4", "aac"),
        ] {
            let path = directory.join(format!("tone.{extension}"));
            let result = std::process::Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=44100:duration=0.5",
                    "-ac",
                    "2",
                    "-c:a",
                    codec,
                    "-strict",
                    "experimental",
                ])
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let samples = read_audio(&path).unwrap();
            assert!(
                (7600..9600).contains(&samples.len()),
                "{extension}: {} samples",
                samples.len()
            );
            assert!(
                samples.iter().any(|sample| sample.abs() > 0.01),
                "{extension}: silent output"
            );
            // A renamed file should decode identically.
            let renamed = directory.join("audio.bin");
            std::fs::rename(&path, &renamed).unwrap();
            assert_eq!(read_audio(&renamed).unwrap(), samples);
            std::fs::remove_file(renamed).unwrap();
        }
        let invalid = directory.join("invalid.mp3");
        std::fs::write(&invalid, b"not audio").unwrap();
        assert!(read_audio(&invalid).is_err());
        std::fs::remove_file(invalid).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn wav_integer_depths_are_scaled_and_mixed_to_mono() {
        for bits in [8, 16, 24, 32] {
            let path =
                std::env::temp_dir().join(format!("pyzh-wav-{}-{bits}.wav", std::process::id()));
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 16_000,
                bits_per_sample: bits,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(&path, spec).unwrap();
            let half = 1_i32 << (bits - 2);
            for sample in [half, half, -half, half] {
                writer.write_sample(sample).unwrap();
            }
            writer.finalize().unwrap();
            assert_eq!(read_audio(&path).unwrap(), vec![0.5, 0.0]);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn model_override_wins_without_changing_other_settings() {
        let arguments = Arguments::try_parse_from([
            "pyzhscriber",
            "transcribe",
            "test.wav",
            "--mode",
            "routed",
            "--model",
            "test-model",
            "--json",
        ])
        .unwrap();
        let Some(Command::Transcribe {
            options,
            json: true,
            standalone: false,
            ..
        }) = arguments.command
        else {
            panic!("expected an app transcription request");
        };
        let mut settings = AppSettings {
            allowed_languages: vec!["ru".into()],
            ..AppSettings::default()
        };
        options.apply(&mut settings);
        assert_eq!(settings.transcription_mode, TranscriptionMode::Single);
        assert_eq!(settings.selected_model, "test-model");
        assert!(settings.allowed_languages.is_empty());
        assert_eq!(settings.routes, AppSettings::default().routes);
    }
}
