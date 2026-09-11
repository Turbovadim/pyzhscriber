//! Compare engine builds with the same WAV manifest and seed.
//! cargo run --release --example whisper_bench -- MODEL.gguf MANIFEST.txt [REPEATS]
//! Each manifest line is a 16 kHz mono PCM16 WAV path. Results are JSON Lines.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde_json::json;
use transcribe_cpp::{
    Backend, KvType, Model, ModelOptions, RunExtension, RunOptions, SessionOptions, Task,
    TimestampKind, WhisperRunOptions,
};

#[derive(Clone, Copy, ValueEnum)]
enum Timestamps {
    Auto,
    None,
    Segment,
}

#[derive(Parser)]
struct Arguments {
    model: PathBuf,
    manifest: PathBuf,
    #[arg(default_value_t = 1)]
    repeats: usize,
    #[arg(long)]
    language: Option<String>,
    /// Comma-separated language codes. An empty value allows all languages.
    #[arg(long, default_value = "en,ru")]
    allowed_languages: String,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    translate: bool,
    #[arg(long, value_enum, default_value = "auto")]
    timestamps: Timestamps,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    #[arg(long, default_value_t = 0)]
    threads: i32,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    kv_f32: bool,
    #[arg(long)]
    temperature: Option<f32>,
    /// Exercise a reused session. The app normally releases it after each run.
    #[arg(long)]
    reuse_session: bool,
}

#[cfg(target_os = "macos")]
fn memory_usage() -> Option<serde_json::Value> {
    // Darwin's rusage_info_v0 layout, from sys/resource.h. This reports the
    // current footprint as well as RSS, which includes reclaimable mappings.
    #[repr(C)]
    #[derive(Default)]
    struct Usage {
        uuid: [u8; 16],
        user_time: u64,
        system_time: u64,
        idle_wakeups: u64,
        interrupt_wakeups: u64,
        pageins: u64,
        wired_size: u64,
        resident_size: u64,
        physical_footprint: u64,
        process_start: u64,
        process_exit: u64,
    }
    #[link(name = "proc")]
    unsafe extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut Usage) -> i32;
    }
    let mut usage = Usage::default();
    // SAFETY: flavor 0 writes exactly rusage_info_v0 into the supplied buffer.
    let status = unsafe { proc_pid_rusage(std::process::id() as i32, 0, &mut usage) };
    (status == 0).then(|| {
        json!({
            "rss_bytes": usage.resident_size,
            "physical_footprint_bytes": usage.physical_footprint,
        })
    })
}

#[cfg(not(target_os = "macos"))]
fn memory_usage() -> Option<serde_json::Value> {
    None
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1
            && spec.sample_rate == 16_000
            && spec.bits_per_sample == 16
            && spec.sample_format == hound::SampleFormat::Int,
        "expected 16 kHz mono PCM16 WAV: {}",
        path.display()
    );
    reader
        .samples::<i16>()
        .map(|sample| Ok(sample? as f32 / i16::MAX as f32))
        .collect()
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    ensure!(arguments.repeats > 0, "repeats must be positive");
    let manifest = fs::read_to_string(&arguments.manifest)?;
    let paths: Vec<_> = manifest.lines().filter(|line| !line.is_empty()).collect();
    ensure!(!paths.is_empty(), "WAV manifest is empty");

    transcribe_cpp::init_backends_default()?;
    let start = Instant::now();
    let model = Model::load_with(
        &arguments.model,
        &ModelOptions {
            backend: if arguments.cpu {
                Backend::Cpu
            } else {
                Backend::Auto
            },
            ..ModelOptions::default()
        },
    )?;
    ensure!(model.arch() == "whisper", "expected a Whisper model");
    println!(
        "{}",
        json!({"event": "load", "model": arguments.model, "backend": model.backend(),
            "load_ms": start.elapsed().as_secs_f64() * 1000.0, "memory": memory_usage()})
    );

    // Match the app's default run settings and release session buffers after
    // each recording. A fixed seed also makes temperature fallback comparable.
    let options = RunOptions {
        language: arguments.language,
        task: if arguments.translate {
            Task::Translate
        } else {
            Task::Transcribe
        },
        target_language: arguments.translate.then(|| "en".into()),
        timestamps: match arguments.timestamps {
            Timestamps::Auto => TimestampKind::Auto,
            Timestamps::None => TimestampKind::None,
            Timestamps::Segment => TimestampKind::Segment,
        },
        family: Some(RunExtension::Whisper(WhisperRunOptions {
            allowed_languages: Some(
                arguments
                    .allowed_languages
                    .split(',')
                    .filter(|code| !code.is_empty())
                    .map(str::to_owned)
                    .collect(),
            ),
            initial_prompt: arguments.prompt,
            seed: Some(arguments.seed),
            temperature: arguments.temperature,
            ..WhisperRunOptions::default()
        })),
        ..RunOptions::default()
    };
    let session_options = SessionOptions {
        n_threads: arguments.threads,
        kv_type: if arguments.kv_f32 {
            KvType::F32
        } else {
            KvType::Auto
        },
        ..SessionOptions::default()
    };
    let mut reused_session = arguments
        .reuse_session
        .then(|| model.session_with(&session_options))
        .transpose()?;
    for repetition in 0..arguments.repeats {
        for path in &paths {
            let pcm = read_wav(Path::new(path)).with_context(|| format!("read {path}"))?;
            let start = Instant::now();
            let result = if let Some(session) = reused_session.as_mut() {
                session.run(&pcm, &options)
            } else {
                let mut session = model.session_with(&session_options)?;
                session.run(&pcm, &options)
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            match result {
                Ok(result) => println!(
                    "{}",
                    json!({"event": "run", "file": path, "repetition": repetition,
                        "audio_ms": pcm.len() as f64 / 16.0, "elapsed_ms": elapsed_ms,
                        "mel_ms": result.timings.mel_ms, "encode_ms": result.timings.encode_ms,
                        "decode_ms": result.timings.decode_ms, "language": result.language,
                        "memory_after_session": memory_usage(),
                        "text": result.text, "raw_text": result.raw_text,
                        "segments": result.segments.iter().map(|s| json!({
                            "start_ms": s.t0_ms, "end_ms": s.t1_ms, "text": s.text
                        })).collect::<Vec<_>>()})
                ),
                Err(error) => {
                    println!(
                        "{}",
                        json!({"event": "error", "file": path, "repetition": repetition,
                            "elapsed_ms": elapsed_ms, "error": error.to_string()})
                    );
                    return Err(error.into());
                }
            }
        }
    }
    drop(reused_session);
    drop(model);
    println!("{}", json!({"event": "unload", "memory": memory_usage()}));
    Ok(())
}
