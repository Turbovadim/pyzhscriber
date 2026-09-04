//! Appends app logs to ~/Library/Logs/Pyzhscriber.log and mirrors them to stderr.

use log::{Level, LevelFilter, Log, Metadata, Record};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

struct FileLogger {
    file: Option<Mutex<File>>,
}

impl Log for FileLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info && metadata.target().starts_with("pyzhscriber")
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!(
            "{} {:<5} {}: {}\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            record.level(),
            record.target(),
            record.args()
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
        if let Some(file) = &self.file {
            let _ = file.lock().write_all(line.as_bytes());
        }
    }

    fn flush(&self) {}
}

pub fn log_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join("Library/Logs").join("Pyzhscriber.log"))
}

pub fn init() {
    let file = log_path().and_then(|path| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(Mutex::new)
    });
    let _ = log::set_boxed_logger(Box::new(FileLogger { file }));
    log::set_max_level(LevelFilter::Info);
}
