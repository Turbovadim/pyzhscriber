//! One JSON request and response per connection on a private Unix socket.
use crate::cli::TranscribeOptions;
use crate::controller::AppController;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Status,
    Transcribe {
        file: PathBuf,
        options: TranscribeOptions,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Status {
        version: String,
        status: String,
        loaded_model: Option<String>,
        /// Physical footprint of the app, the number Activity Monitor shows.
        #[serde(default)]
        app_memory: Option<u64>,
        /// Size of the loaded weights file. It is memory-mapped, so macOS
        /// counts it as Cached Files instead of the app's footprint.
        #[serde(default)]
        model_weights: Option<u64>,
    },
    Transcript {
        text: String,
    },
    Error {
        error: String,
    },
}

fn socket_path() -> Result<PathBuf> {
    Ok(dirs::data_dir()
        .context("application data directory is unavailable")?
        .join(env!("PYZHSCRIBER_BUNDLE_ID"))
        .join("cli/rpc.sock"))
}

pub fn request(request: &Request) -> Result<Response> {
    let mut stream = UnixStream::connect(socket_path()?).context(
        "cannot connect to Pyzhscriber; open an updated app first, or use transcribe --standalone",
    )?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    // Transcription may take a long time. EOF still reports an app exit promptly.
    write_message(&mut stream, request)?;
    read_message(&mut stream, 16 * 1024 * 1024)
}

pub struct Server {
    path: PathBuf,
    stopped: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Server {
    pub fn start(controller: &Arc<AppController>) -> Result<Self> {
        Self::start_at(controller, socket_path()?)
    }

    fn start_at(controller: &Arc<AppController>, path: PathBuf) -> Result<Self> {
        let directory = path.parent().context("socket has no parent")?;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(directory)?;
        anyhow::ensure!(
            !directory.symlink_metadata()?.file_type().is_symlink(),
            "CLI directory cannot be a symlink"
        );
        std::fs::set_permissions(directory, std::fs::Permissions::from_mode(0o700))?;
        if path.exists() {
            anyhow::ensure!(
                path.symlink_metadata()?.file_type().is_socket(),
                "CLI socket path contains another file"
            );
            anyhow::ensure!(
                UnixStream::connect(&path).is_err(),
                "CLI socket is already in use"
            );
            std::fs::remove_file(&path)?;
        }
        let listener = UnixListener::bind(&path).context("bind CLI socket")?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        let stopped = Arc::new(AtomicBool::new(false));
        let stop = stopped.clone();
        let controller = Arc::downgrade(controller);
        // Fixed workers and a bounded queue prevent idle clients spawning unlimited threads.
        let (sender, receiver) = std::sync::mpsc::sync_channel::<UnixStream>(8);
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        for _ in 0..2 {
            let receiver = receiver.clone();
            let controller = controller.clone();
            let stop = stopped.clone();
            std::thread::spawn(move || {
                loop {
                    let Ok(mut stream) = receiver.lock().unwrap().recv() else {
                        break;
                    };
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Some(controller) = controller.upgrade() else {
                        break;
                    };
                    let result = (|| -> Result<Response> {
                        stream.set_read_timeout(Some(Duration::from_secs(5)))?;
                        stream.set_write_timeout(Some(Duration::from_secs(5)))?;
                        let request = read_message(&mut stream, 64 * 1024)?;
                        handle(&controller, request)
                    })();
                    let response = result.unwrap_or_else(|error| Response::Error {
                        error: format!("{error:#}"),
                    });
                    let _ = write_message(&mut stream, &response);
                }
            });
        }
        let thread = std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                match stream {
                    Ok(mut stream) => {
                        if let Err(std::sync::mpsc::TrySendError::Full(returned)) =
                            sender.try_send(stream)
                        {
                            stream = returned;
                            let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
                            let _ = write_message(
                                &mut stream,
                                &Response::Error {
                                    error: "CLI request queue is full; retry later".into(),
                                },
                            );
                        }
                    }
                    Err(error) => {
                        log::warn!("CLI accept failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            path,
            stopped,
            thread: Some(thread),
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::SeqCst);
        let _ = UnixStream::connect(&self.path);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        let _ = std::fs::remove_file(&self.path);
    }
}

fn handle(controller: &AppController, request: Request) -> Result<Response> {
    match request {
        Request::Status => {
            let loaded_model = controller.loaded_model_id();
            let model_weights = loaded_model.as_deref().and_then(|id| {
                let path = controller.models.resolve_path(id).ok()?;
                Some(std::fs::metadata(path).ok()?.len())
            });
            Ok(Response::Status {
                version: env!("CARGO_PKG_VERSION").into(),
                status: controller.status().label().to_lowercase(),
                loaded_model,
                app_memory: physical_footprint(),
                model_weights,
            })
        }
        Request::Transcribe { file, options } => Ok(Response::Transcript {
            text: controller.transcribe_file(&file, &options)?,
        }),
    }
}

#[cfg(target_os = "macos")]
fn physical_footprint() -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::uninit();
    // SAFETY: RUSAGE_INFO_V2 tells the kernel to fill exactly one rusage_info_v2.
    let result = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V2,
            info.as_mut_ptr().cast(),
        )
    };
    // SAFETY: a zero result means the kernel initialized the whole struct.
    (result == 0).then(|| unsafe { info.assume_init() }.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
fn physical_footprint() -> Option<u64> {
    None
}

fn write_message(stream: &mut UnixStream, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *stream, value)?;
    stream.write_all(b"\n")?;
    Ok(())
}

fn read_message<T: serde::de::DeserializeOwned>(stream: &mut UnixStream, limit: u64) -> Result<T> {
    let mut data = Vec::new();
    BufReader::new(stream.take(limit + 1)).read_until(b'\n', &mut data)?;
    if data.len() as u64 > limit {
        bail!("CLI message exceeds size limit");
    }
    anyhow::ensure!(
        data.last() == Some(&b'\n'),
        "incomplete CLI response or request"
    );
    Ok(serde_json::from_slice(&data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CLI_TEST_BINARY, CLI_TEST_AUDIO, CLI_TEST_MODEL and a free app socket"]
    fn cli_transcribes_through_resident_model() {
        let binary = std::env::var("CLI_TEST_BINARY").unwrap();
        let audio = std::env::var("CLI_TEST_AUDIO").unwrap();
        let model = std::env::var("CLI_TEST_MODEL").unwrap();
        let controller = AppController::new().unwrap();
        let server = Server::start(&controller).unwrap();
        let run = |args: &[&str]| {
            let output = std::process::Command::new(&binary)
                .args(args)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice::<Response>(&output.stdout).unwrap()
        };
        for _ in 0..2 {
            assert!(
                matches!(run(&["transcribe", &audio, "--model", &model, "--json"]),
                Response::Transcript { text } if !text.trim().is_empty())
            );
            assert!(matches!(run(&["status", "--json"]),
                Response::Status { loaded_model: Some(loaded), .. } if loaded == model));
        }
        drop(server);
        controller.shutdown();
    }

    #[test]
    fn server_status_errors_and_socket_lifecycle() {
        let directory = std::env::temp_dir().join(format!("pyzh-cli-test-{}", std::process::id()));
        let path = directory.join("rpc.sock");
        let controller = AppController::new().unwrap();
        let server = Server::start_at(&controller, path.clone()).unwrap();
        assert_eq!(
            directory.metadata().unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(path.metadata().unwrap().permissions().mode() & 0o777, 0o600);
        assert!(Server::start_at(&controller, path.clone()).is_err());
        let mut stream = UnixStream::connect(&path).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_message(&mut stream, &Request::Status).unwrap();
        assert!(matches!(
            read_message(&mut stream, 4096).unwrap(),
            Response::Status { app_memory, .. } if app_memory.is_some() == cfg!(target_os = "macos")
        ));
        let mut stream = UnixStream::connect(&path).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        write_message(
            &mut stream,
            &Request::Transcribe {
                file: directory.join("missing.wav"),
                options: TranscribeOptions::default(),
            },
        )
        .unwrap();
        assert!(
            matches!(read_message(&mut stream, 4096).unwrap(), Response::Error { error } if error.contains("regular audio"))
        );
        drop(server);
        assert!(!path.exists());
        drop(UnixListener::bind(&path).unwrap());
        drop(Server::start_at(&controller, path.clone()).unwrap());
        assert!(!path.exists());
        controller.shutdown();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn socket_round_trip_and_limits() {
        let (mut client, mut server) = UnixStream::pair().unwrap();
        write_message(&mut client, &Request::Status).unwrap();
        assert!(matches!(
            read_message(&mut server, 1024).unwrap(),
            Request::Status
        ));
        write_message(
            &mut server,
            &Response::Transcript {
                text: "Привет\nworld".into(),
            },
        )
        .unwrap();
        assert!(
            matches!(read_message(&mut client, 1024).unwrap(), Response::Transcript { text } if text == "Привет\nworld")
        );
        client.write_all(b"123456789\n").unwrap();
        assert!(
            read_message::<Request>(&mut server, 4)
                .unwrap_err()
                .to_string()
                .contains("size limit")
        );
    }
}
