use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct HistoryEntry {
    pub id: i64,
    pub text: String,
    pub model_id: String,
    pub language: Option<String>,
    pub audio_seconds: f64,
    pub created_at: DateTime<Utc>,
}

pub struct HistoryStore {
    connection: Mutex<Connection>,
    recordings_dir: PathBuf,
}

impl HistoryStore {
    pub fn new() -> Result<Self> {
        let data = dirs::data_dir()
            .context("application data directory is unavailable")?
            .join(env!("PYZHSCRIBER_BUNDLE_ID"));
        fs::create_dir_all(&data)?;
        let recordings = data.join("recordings");
        fs::create_dir_all(&recordings)?;
        let connection = Connection::open(data.join("history.db"))?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                model_id TEXT NOT NULL,
                language TEXT,
                audio_seconds REAL NOT NULL,
                created_at TEXT NOT NULL
            );",
        )?;
        Ok(Self {
            connection: Mutex::new(connection),
            recordings_dir: recordings,
        })
    }

    pub fn add(
        &self,
        text: &str,
        model_id: &str,
        language: Option<&str>,
        audio_seconds: f64,
    ) -> Result<HistoryEntry> {
        let now = Utc::now();
        let connection = self.connection.lock();
        connection.execute(
            "INSERT INTO history(text, model_id, language, audio_seconds, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![text, model_id, language, audio_seconds, now.to_rfc3339()],
        )?;
        Ok(HistoryEntry {
            id: connection.last_insert_rowid(),
            text: text.to_string(),
            model_id: model_id.to_string(),
            language: language.map(str::to_string),
            audio_seconds,
            created_at: now,
        })
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<HistoryEntry>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT id, text, model_id, language, audio_seconds, created_at
             FROM history ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = statement.query_map([limit as i64], |row| {
            let created_at: String = row.get(5)?;
            Ok(HistoryEntry {
                id: row.get(0)?,
                text: row.get(1)?,
                model_id: row.get(2)?,
                language: row.get(3)?,
                audio_seconds: row.get(4)?,
                created_at: DateTime::parse_from_rfc3339(&created_at)
                    .map(|value| value.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
            })
        })?;
        Ok(rows.filter_map(Result::ok).collect())
    }

    pub fn clear(&self) -> Result<()> {
        self.connection.lock().execute("DELETE FROM history", [])?;
        Ok(())
    }

    pub fn save_recording(&self, audio: &[f32]) -> Result<PathBuf> {
        let path = self
            .recordings_dir
            .join(format!("pyzhscriber-{}.wav", Utc::now().timestamp_millis()));
        crate::audio::write_wav(
            audio,
            std::io::BufWriter::new(std::fs::File::create(&path)?),
        )?;
        Ok(path)
    }
}
