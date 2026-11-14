use crate::audio::AudioRecorder;
use crate::cloud::LiveSession;
use crate::history::{HistoryEntry, HistoryStore};
use crate::hotkey::{self, KeyEvent, RecorderStep, Shortcut, ShortcutRecorder};
use crate::models::{DownloadError, ModelChoice, ModelRegistry};
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
    Settings(Box<AppSettings>),
    ShortcutCapture(ShortcutCapture),
    DownloadProgress {
        model_id: String,
        progress: f32,
    },
    /// A download ended without a new model, by failure or cancel.
    DownloadStopped {
        model_id: String,
    },
    ModelsChanged(Vec<ModelChoice>),
}

struct Transcription {
    text: String,
    /// What produced the text, recorded in history.
    source: String,
    /// Set when the cloud failed and a local model stood in.
    cloud_error: Option<anyhow::Error>,
}

pub struct AppController {
    settings: RwLock<AppSettings>,
    /// Parsed from `settings.shortcut` on save, so the global key tap does not
    /// re-parse it for every key typed in any app.
    shortcut: RwLock<Option<Shortcut>>,
    settings_store: SettingsStore,
    pub models: Arc<ModelRegistry>,
    recorder: AudioRecorder,
    /// Fed by the recorder while a streaming cloud model is in use.
    live_session: Mutex<Option<LiveSession>>,
    transcriber: Arc<Transcriber>,
    pub history: HistoryStore,
    status: RwLock<RuntimeStatus>,
    event_tx: async_channel::Sender<RuntimeEvent>,
    event_rx: async_channel::Receiver<RuntimeEvent>,
    shortcut_down: AtomicBool,
    shutting_down: AtomicBool,
    shortcut_listener_started: AtomicBool,
    shortcut_recorder: Mutex<Option<ShortcutRecorder>>,
    activity_generation: Arc<AtomicU64>,
    /// Cancel flags of in-flight model downloads, keyed by model id.
    downloads: Mutex<std::collections::HashMap<String, Arc<AtomicBool>>>,
}

impl AppController {
    pub fn new() -> Result<Arc<Self>> {
        let settings_store = SettingsStore::new()?;
        let mut settings = settings_store.load()?;
        let models = Arc::new(ModelRegistry::new()?);
        // Settings may name a non-default quant (e.g. imported from Handy); the
        // app tracks models by their default-file id and resolves the quant.
        settings.selected_model = models.canonical_id(&settings.selected_model);
        for route in &mut settings.routes {
            route.model = models.canonical_id(&route.model);
        }
        let transcriber = Arc::new(Transcriber::new(models.clone()));
        let history = HistoryStore::new()?;
        let (event_tx, event_rx) = async_channel::unbounded();
        Ok(Arc::new(Self {
            shortcut: RwLock::new(parse_shortcut(&settings.shortcut)),
            settings: RwLock::new(settings),
            settings_store,
            models,
            recorder: AudioRecorder::new(),
            live_session: Mutex::new(None),
            transcriber,
            history,
            status: RwLock::new(RuntimeStatus::Idle),
            event_tx,
            event_rx,
            shortcut_down: AtomicBool::new(false),
            shutting_down: AtomicBool::new(false),
            shortcut_listener_started: AtomicBool::new(false),
            shortcut_recorder: Mutex::new(None),
            activity_generation: Arc::new(AtomicU64::new(0)),
            downloads: Mutex::new(std::collections::HashMap::new()),
        }))
    }

    pub fn start(self: &Arc<Self>) {
        self.start_shortcut_listener();
    }

    pub fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::SeqCst);
        self.recorder.cancel();
        self.transcriber.shutdown();
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
        let sensitivity = self.settings.read().overlay_sensitivity.clamp(0.25, 6.0);
        self.recorder
            .levels()
            .map(|level| (level * sensitivity).clamp(0.0, 1.0))
    }

    pub fn save_settings(&self, settings: AppSettings) -> Result<()> {
        let autostart_changed =
            self.settings.read().autostart_enabled != settings.autostart_enabled;
        if autostart_changed {
            startup::set_enabled(settings.autostart_enabled)?;
        }
        self.settings_store.save(&settings)?;
        *self.shortcut.write() = parse_shortcut(&settings.shortcut);
        *self.settings.write() = settings.clone();
        let _ = self
            .event_tx
            .try_send(RuntimeEvent::Settings(Box::new(settings)));
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
        if self.shutting_down.load(Ordering::SeqCst) || self.recorder.is_recording() {
            return;
        }
        let settings = self.settings();
        self.activity_generation.fetch_add(1, Ordering::SeqCst);
        let (session, tap) = if settings.cloud_enabled && crate::cloud::streams(&settings) {
            let (session, tap) = LiveSession::start(&settings);
            (Some(session), Some(tap))
        } else {
            (None, None)
        };
        match self
            .recorder
            .start(settings.selected_microphone.as_deref(), tap)
        {
            Ok(()) => {
                *self.live_session.lock() = session;
                if settings.audio_feedback {
                    sound::play(FeedbackSound::Start);
                }
                log::info!("recording started");
                self.set_status(RuntimeStatus::Recording);
                if !settings.cloud_enabled {
                    let transcriber = self.transcriber.clone();
                    std::thread::spawn(move || {
                        if let Err(error) = transcriber.preload_for(&settings) {
                            log::warn!("model preload failed: {error:#}");
                        }
                    });
                }
            }
            Err(error) => self.set_status(RuntimeStatus::Error(error.to_string())),
        }
    }

    pub fn stop_recording(self: &Arc<Self>) {
        if !self.recorder.is_recording() {
            log::info!("stop requested while no recording was active");
            return;
        }
        let session = self.live_session.lock().take();
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
        std::thread::spawn(move || controller.finish_transcription(audio, session));
    }

    pub fn cancel(&self) {
        self.recorder.cancel();
        self.live_session.lock().take();
        self.set_status(RuntimeStatus::Idle);
    }

    pub fn unload_model(&self) {
        self.transcriber.unload();
    }

    pub fn loaded_model_id(&self) -> Option<String> {
        self.transcriber.loaded_model_id()
    }

    /// CLI jobs reuse the model, but never insert text or change saved settings/history.
    pub fn transcribe_file(
        &self,
        file: &std::path::Path,
        options: &crate::cli::TranscribeOptions,
    ) -> Result<String> {
        anyhow::ensure!(
            !self.shutting_down.load(Ordering::SeqCst),
            "app is shutting down"
        );
        let mut settings = self.settings();
        options.apply(&mut settings);
        let audio = crate::cli::read_audio(file)?;
        self.activity_generation.fetch_add(1, Ordering::SeqCst);
        let result = self
            .transcribe_audio(&audio, &settings, None)
            .map(|done| done.text);
        if !self.recorder.is_recording() {
            self.schedule_model_unload(settings.model_unload_minutes);
        }
        result
    }

    /// Starts a background download; a second request for the same model
    /// while one is running is ignored.
    pub fn download_model(self: &Arc<Self>, model_id: String) {
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut downloads = self.downloads.lock();
            if downloads.contains_key(&model_id) {
                return;
            }
            downloads.insert(model_id.clone(), cancel.clone());
        }
        let controller = self.clone();
        std::thread::spawn(move || {
            let sender = controller.event_tx.clone();
            let id_for_progress = model_id.clone();
            let result = controller
                .models
                .download(&model_id, &cancel, move |progress| {
                    let _ = sender.try_send(RuntimeEvent::DownloadProgress {
                        model_id: id_for_progress.clone(),
                        progress,
                    });
                });
            controller.downloads.lock().remove(&model_id);
            match result {
                Ok(_) => {}
                Err(DownloadError::Cancelled) => {
                    log::info!("download of {model_id} cancelled");
                    let _ = controller
                        .event_tx
                        .try_send(RuntimeEvent::DownloadStopped { model_id });
                }
                Err(DownloadError::Failed(error)) => {
                    let _ = controller
                        .event_tx
                        .try_send(RuntimeEvent::DownloadStopped { model_id });
                    controller.set_status(RuntimeStatus::Error(format!("{error:#}")));
                }
            }
            controller.refresh_models();
        });
    }

    /// Stops a running download, keeping its partial file for a later resume.
    pub fn cancel_download(&self, model_id: &str) {
        if let Some(cancel) = self.downloads.lock().get(model_id) {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Deletes a model's files from our models folder. Loaded models are
    /// dropped first, since the removed file may back the transcriber or the
    /// language detector; they reload on next use.
    pub fn remove_model(&self, model_id: &str) -> Result<()> {
        self.transcriber.unload();
        let result = self.models.remove(model_id);
        self.refresh_models();
        result
    }

    /// Re-reads install state from disk, picking up models added elsewhere.
    pub fn refresh_models(&self) {
        let _ = self
            .event_tx
            .try_send(RuntimeEvent::ModelsChanged(self.models.choices()));
    }

    /// `session` has already received this audio while it was recorded.
    fn finish_transcription(&self, audio: Vec<f32>, session: Option<LiveSession>) {
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
        let result = self
            .transcribe_audio(&audio, &settings, session)
            .and_then(|done| {
                let mut text = post_process::clean(&done.text, &settings)?;
                if self.shutting_down.load(Ordering::SeqCst) || text.trim().is_empty() {
                    return Ok((None, done));
                }
                // Added after cleanup, which re-joins words and would drop it.
                if settings.append_trailing_space {
                    text.push(' ');
                }
                output::deliver(&text, &settings)?;
                Ok((Some(text), done))
            });

        match result {
            Ok((Some(text), done)) => {
                // The text is already in place, so the pill leaves before the
                // bookkeeping below rather than after it.
                match done.cloud_error {
                    Some(error) => self.set_status(RuntimeStatus::Error(format!(
                        "Cloud dictation failed, so a local model was used: {error:#}"
                    ))),
                    None => self.set_status(RuntimeStatus::Idle),
                }
                self.schedule_model_unload(settings.model_unload_minutes);
                let audio_seconds = audio.len() as f64 / 16_000.0;
                log::info!(
                    "transcribed {audio_seconds:.1}s of audio into {} characters",
                    text.chars().count()
                );
                match self
                    .history
                    .add(text.trim_end(), &done.source, None, audio_seconds)
                {
                    Ok(entry) => {
                        let _ = self.event_tx.try_send(RuntimeEvent::Transcript(entry));
                    }
                    Err(error) => log::error!("failed to save transcript history: {error:#}"),
                }
                if let Err(error) = self.history.save_recording(&audio) {
                    log::warn!("failed to save recording: {error:#}");
                }
            }
            Ok((None, _)) => {
                log::info!("transcription produced no text");
                self.set_status(RuntimeStatus::Idle);
                self.schedule_model_unload(settings.model_unload_minutes);
            }
            Err(error) => self.set_status(RuntimeStatus::Error(format!("{error:#}"))),
        }
    }

    /// Transcribes through the cloud when enabled, falling back to local
    /// models so a network or billing problem does not lose the dictation.
    /// When both fail, the cloud error is the one worth reporting. A live
    /// session already holds the audio, so only its transcript is awaited.
    fn transcribe_audio(
        &self,
        audio: &[f32],
        settings: &AppSettings,
        session: Option<LiveSession>,
    ) -> Result<Transcription> {
        let local_source = || match settings.transcription_mode {
            TranscriptionMode::Single => settings.selected_model.clone(),
            TranscriptionMode::Routed => {
                format!("automatic {}-language routing", settings.routes.len())
            }
        };
        if !settings.cloud_enabled {
            return Ok(Transcription {
                text: self.transcriber.transcribe(audio, settings)?,
                source: local_source(),
                cloud_error: None,
            });
        }
        let cloud = match session {
            Some(session) => session.finish(),
            None => crate::cloud::transcribe(audio, settings),
        };
        match cloud {
            Ok(text) => Ok(Transcription {
                text,
                source: crate::cloud::source(settings),
                cloud_error: None,
            }),
            Err(cloud_error) => {
                log::warn!("cloud transcription failed, trying local models: {cloud_error:#}");
                match self.transcriber.transcribe(audio, settings) {
                    Ok(text) => Ok(Transcription {
                        text,
                        source: local_source(),
                        cloud_error: Some(cloud_error),
                    }),
                    Err(local_error) => {
                        log::warn!("local fallback failed: {local_error:#}");
                        Err(cloud_error)
                    }
                }
            }
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
        if self.shutting_down.load(Ordering::SeqCst) {
            return;
        }
        if self.feed_shortcut_recorder(event) {
            return;
        }
        let Some(shortcut) = *self.shortcut.read() else {
            return;
        };
        let push_to_talk = self.settings.read().push_to_talk;
        if shortcut.is_pressed(event) {
            if !self.shortcut_down.swap(true, Ordering::SeqCst) {
                log::info!("shortcut {shortcut} pressed");
                if push_to_talk {
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
            log::info!("shortcut {shortcut} released");
            if push_to_talk {
                self.stop_recording();
            }
        }
    }
}

fn parse_shortcut(binding: &str) -> Option<Shortcut> {
    let shortcut = Shortcut::parse(binding);
    if shortcut.is_none() {
        log::warn!("unknown shortcut binding {binding:?}");
    }
    shortcut
}
