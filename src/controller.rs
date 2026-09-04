use crate::audio::AudioRecorder;
use crate::history::{HistoryEntry, HistoryStore};
use crate::hotkey::{self, KeyEvent, RecorderStep, Shortcut, ShortcutRecorder};
use crate::models::{ModelChoice, ModelRegistry};
use crate::output;
use crate::post_process;
use crate::settings::{AppSettings, SettingsStore, TranscriptionMode};
use crate::sound::{self, FeedbackSound};
use crate::startup;
use crate::transcription::Transcriber;
use crate::vad;
use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    Idle,
    Recording,
    Transcribing,
    Error(String),
}

impl RuntimeStatus {
    pub fn label(&self) -> &str {
        match self {
            Self::Idle => "Ready",
            Self::Recording => "Recording",
            Self::Transcribing => "Transcribing",
            Self::Error(_) => "Error",
        }
    }
}

/// Progress of recording a new transcription shortcut, shown live in settings.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum ShortcutCapture {
    #[default]
    Idle,
    /// Carries the modifiers currently held, e.g. `⌥ ⌘`.
    Listening(String),
    NeedsModifier,
}

#[derive(Clone, Debug)]
pub enum RuntimeEvent {
    Status(RuntimeStatus),
    Transcript(HistoryEntry),
    Settings(AppSettings),
    ShortcutCapture(ShortcutCapture),
    DownloadProgress { model_id: String, progress: f32 },
    ModelsChanged(Vec<ModelChoice>),
}

pub struct AppController {
    settings: RwLock<AppSettings>,
    settings_store: SettingsStore,
    pub models: Arc<ModelRegistry>,
    recorder: AudioRecorder,
    transcriber: Arc<Transcriber>,
    pub history: HistoryStore,
    status: RwLock<RuntimeStatus>,
    event_tx: async_channel::Sender<RuntimeEvent>,
    event_rx: async_channel::Receiver<RuntimeEvent>,
    shortcut_down: AtomicBool,
    shortcut_listener_started: AtomicBool,
    shortcut_recorder: Mutex<Option<ShortcutRecorder>>,
    activity_generation: Arc<AtomicU64>,
}

impl AppController {
    pub fn new() -> Result<Arc<Self>> {
        let settings_store = SettingsStore::new()?;
        let settings = settings_store.load()?;
        let models = Arc::new(ModelRegistry::new()?);
        let transcriber = Arc::new(Transcriber::new(models.clone()));
        let history = HistoryStore::new()?;
        let (event_tx, event_rx) = async_channel::unbounded();
        Ok(Arc::new(Self {
            settings: RwLock::new(settings),
            settings_store,
            models,
            recorder: AudioRecorder::new(),
            transcriber,
            history,
            status: RwLock::new(RuntimeStatus::Idle),
            event_tx,
            event_rx,
            shortcut_down: AtomicBool::new(false),
            shortcut_listener_started: AtomicBool::new(false),
            shortcut_recorder: Mutex::new(None),
            activity_generation: Arc::new(AtomicU64::new(0)),
        }))
    }

    pub fn start(self: &Arc<Self>) {
        self.start_shortcut_listener();
    }

    pub fn events(&self) -> async_channel::Receiver<RuntimeEvent> {
        self.event_rx.clone()
    }

    pub fn settings(&self) -> AppSettings {
        self.settings.read().clone()
    }

    pub fn status(&self) -> RuntimeStatus {
        self.status.read().clone()
    }

    pub fn mic_levels(&self) -> [f32; 9] {
        self.recorder.levels()
    }

    pub fn save_settings(&self, settings: AppSettings) -> Result<()> {
        let autostart_changed =
            self.settings.read().autostart_enabled != settings.autostart_enabled;
        if autostart_changed {
            startup::set_enabled(settings.autostart_enabled)?;
        }
        self.settings_store.save(&settings)?;
        *self.settings.write() = settings.clone();
        let _ = self.event_tx.try_send(RuntimeEvent::Settings(settings));
        Ok(())
    }

    /// Routes the next key presses into a shortcut recorder instead of the
    /// transcription shortcut. Needs the global tap, which Input Monitoring gates.
    pub fn begin_shortcut_capture(&self) -> Result<()> {
        if !self.shortcut_listener_started.load(Ordering::SeqCst) {
            anyhow::bail!("Grant Input Monitoring first so the shortcut can be detected");
        }
        self.shortcut_down.store(false, Ordering::SeqCst);
        *self.shortcut_recorder.lock() = Some(ShortcutRecorder::default());
        let _ = self
            .event_tx
            .try_send(RuntimeEvent::ShortcutCapture(ShortcutCapture::Listening(
                String::new(),
            )));
        Ok(())
    }

    pub fn cancel_shortcut_capture(&self) {
        if self.shortcut_recorder.lock().take().is_some() {
            let _ = self
                .event_tx
                .try_send(RuntimeEvent::ShortcutCapture(ShortcutCapture::Idle));
        }
    }

    /// Returns true when the event was consumed by an active capture.
    fn feed_shortcut_recorder(&self, event: KeyEvent) -> bool {
        let mut recorder = self.shortcut_recorder.lock();
        let Some(active) = recorder.as_mut() else {
            return false;
        };
        let capture = match active.feed(event) {
            RecorderStep::Listening(held) => ShortcutCapture::Listening(held),
            RecorderStep::NeedsModifier => ShortcutCapture::NeedsModifier,
            RecorderStep::Captured(shortcut) => {
                *recorder = None;
                drop(recorder);
                let mut settings = self.settings();
                settings.shortcut = shortcut.to_string();
                log::info!("shortcut changed to {}", settings.shortcut);
                if let Err(error) = self.save_settings(settings) {
                    self.set_status(RuntimeStatus::Error(format!("{error:#}")));
                }
                ShortcutCapture::Idle
            }
            RecorderStep::Cancelled => {
                *recorder = None;
                ShortcutCapture::Idle
            }
        };
        let _ = self
            .event_tx
            .try_send(RuntimeEvent::ShortcutCapture(capture));
        true
    }

    pub fn toggle_recording(self: &Arc<Self>) {
        if self.recorder.is_recording() {
            self.stop_recording();
        } else {
            self.start_recording();
        }
    }

    pub fn start_recording(self: &Arc<Self>) {
        if self.recorder.is_recording() {
            return;
        }
        let settings = self.settings();
        self.activity_generation.fetch_add(1, Ordering::SeqCst);
        match self.recorder.start(settings.selected_microphone.as_deref()) {
            Ok(()) => {
                if settings.audio_feedback {
                    sound::play(FeedbackSound::Start);
                }
                log::info!("recording started");
                self.set_status(RuntimeStatus::Recording);
                let transcriber = self.transcriber.clone();
                std::thread::spawn(move || {
                    if let Err(error) = transcriber.preload_for(&settings) {
                        log::warn!("model preload failed: {error:#}");
                    }
                });
            }
            Err(error) => self.set_status(RuntimeStatus::Error(error.to_string())),
        }
    }

    pub fn stop_recording(self: &Arc<Self>) {
        if !self.recorder.is_recording() {
            log::info!("stop requested while no recording was active");
            return;
        }
        let audio = match self.recorder.stop() {
            Ok(audio) => audio,
            Err(error) => {
                self.set_status(RuntimeStatus::Error(error.to_string()));
                return;
            }
        };
        if self.settings().audio_feedback {
            sound::play(FeedbackSound::Stop);
        }
        self.set_status(RuntimeStatus::Transcribing);
        let controller = self.clone();
        std::thread::spawn(move || controller.finish_transcription(audio));
    }

    pub fn cancel(&self) {
        self.recorder.cancel();
        self.set_status(RuntimeStatus::Idle);
    }

    pub fn unload_model(&self) {
        self.transcriber.unload();
    }

    pub fn loaded_model_id(&self) -> Option<String> {
        self.transcriber.loaded_model_id()
    }

    pub fn download_model(self: &Arc<Self>, model_id: String) {
        let controller = self.clone();
        std::thread::spawn(move || {
            let id_for_progress = model_id.clone();
            let sender = controller.event_tx.clone();
            let result = controller.models.download(&model_id, move |progress| {
                let _ = sender.try_send(RuntimeEvent::DownloadProgress {
                    model_id: id_for_progress.clone(),
                    progress,
                });
            });
            match result {
                Ok(_) => {
                    let _ = controller.event_tx.try_send(RuntimeEvent::ModelsChanged(
                        controller.models.default_choices(),
                    ));
                }
                Err(error) => controller.set_status(RuntimeStatus::Error(error.to_string())),
            }
        });
    }

    fn finish_transcription(&self, audio: Vec<f32>) {
        let settings = self.settings();
        let captured = audio.len();
        let mut audio = if settings.vad_enabled {
            vad::trim_silence(&audio)
        } else {
            audio
        };
        if audio.is_empty() {
            log::info!(
                "discarded {:.1}s recording: no speech detected",
                captured as f64 / 16_000.0
            );
            self.set_status(RuntimeStatus::Idle);
            return;
        }
        let peak = crate::audio::normalize_peak(&mut audio);
        log::info!(
            "captured {:.1}s (peak {peak:.4}), transcribing {:.1}s",
            captured as f64 / 16_000.0,
            audio.len() as f64 / 16_000.0
        );
        let model_id = match settings.transcription_mode {
            TranscriptionMode::Single => settings.selected_model.clone(),
            TranscriptionMode::Routed => {
                format!("automatic {}-language routing", settings.routes.len())
            }
        };
        let result = self
            .transcriber
            .transcribe(&audio, &settings)
            .and_then(|text| post_process::clean(&text, &settings))
            .and_then(|text| {
                if text.trim().is_empty() {
                    return Ok(None);
                }
                output::deliver(&text, &settings)?;
                let entry =
                    self.history
                        .add(&text, &model_id, None, audio.len() as f64 / 16_000.0)?;
                let _ = self.history.save_recording(&audio);
                Ok(Some(entry))
            });

        match result {
            Ok(Some(entry)) => {
                log::info!(
                    "transcribed {:.1}s of audio into {} characters",
                    entry.audio_seconds,
                    entry.text.chars().count()
                );
                let _ = self.event_tx.try_send(RuntimeEvent::Transcript(entry));
                self.set_status(RuntimeStatus::Idle);
                self.schedule_model_unload(settings.model_unload_minutes);
            }
            Ok(None) => {
                log::info!("transcription produced no text");
                self.set_status(RuntimeStatus::Idle);
                self.schedule_model_unload(settings.model_unload_minutes);
            }
            Err(error) => self.set_status(RuntimeStatus::Error(format!("{error:#}"))),
        }
    }

    fn set_status(&self, status: RuntimeStatus) {
        if let RuntimeStatus::Error(message) = &status {
            log::error!("{message}");
        }
        *self.status.write() = status.clone();
        let _ = self.event_tx.try_send(RuntimeEvent::Status(status));
    }

    fn schedule_model_unload(&self, minutes: Option<u32>) {
        let Some(minutes) = minutes else {
            return;
        };
        let generation = self.activity_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let transcriber = self.transcriber.clone();
        let activity = self.activity_generation.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(u64::from(minutes) * 60));
            if activity.load(Ordering::SeqCst) == generation {
                transcriber.unload();
            }
        });
    }

    fn start_shortcut_listener(self: &Arc<Self>) {
        if self.shortcut_listener_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        if let Err(error) = hotkey::listen(sender) {
            self.shortcut_listener_started
                .store(false, Ordering::SeqCst);
            log::error!("global shortcut listener failed: {error:#}");
            self.set_status(RuntimeStatus::Error(format!(
                "global shortcut listener failed: {error:#}"
            )));
            return;
        }
        log::info!("global shortcut listener started");
        let controller = self.clone();
        std::thread::spawn(move || {
            for event in receiver {
                controller.handle_key(event);
            }
        });
    }

    fn handle_key(self: &Arc<Self>, event: KeyEvent) {
        if self.feed_shortcut_recorder(event) {
            return;
        }
        let settings = self.settings();
        let Some(shortcut) = Shortcut::parse(&settings.shortcut) else {
            log::warn!("unknown shortcut binding {:?}", settings.shortcut);
            return;
        };
        if shortcut.is_pressed(event) {
            if !self.shortcut_down.swap(true, Ordering::SeqCst) {
                log::info!("shortcut {} pressed", settings.shortcut);
                if settings.push_to_talk {
                    self.start_recording();
                } else {
                    self.toggle_recording();
                }
            }
        } else if event.pressed && event.key_code == hotkey::KEY_ESCAPE {
            if self.recorder.is_recording() {
                log::info!("recording cancelled with Escape");
            }
            self.cancel();
        } else if shortcut.is_released(event) && self.shortcut_down.swap(false, Ordering::SeqCst) {
            log::info!("shortcut {} released", settings.shortcut);
            if settings.push_to_talk {
                self.stop_recording();
            }
        }
    }
}
