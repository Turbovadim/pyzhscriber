use crate::models::ModelRegistry;
use crate::settings::{AppSettings, TranscriptionMode};
use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;
use std::sync::Arc;
use transcribe_cpp::{
    Backend, Model, ModelOptions, RunExtension, RunOptions, Task, WhisperRunOptions,
};

const LANGUAGE_DETECTOR: &str = "handy-computer/whisper-tiny-gguf/whisper-tiny-Q8_0.gguf";
const LANGUAGE_DETECT_SAMPLES: usize = 5 * 16_000;

struct LoadedEngine {
    model_id: String,
    model: Model,
}

pub struct Transcriber {
    registry: Arc<ModelRegistry>,
    loaded: Mutex<Option<LoadedEngine>>,
    last_routed_language: Mutex<Option<String>>,
    last_single_language: Mutex<Option<String>>,
}

impl Transcriber {
    pub fn new(registry: Arc<ModelRegistry>) -> Self {
        transcribe_cpp::init_logging();
        if let Err(error) = transcribe_cpp::init_backends_default() {
            log::warn!("transcribe-cpp backend initialization failed: {error}");
        }
        Self {
            registry,
            loaded: Mutex::new(None),
            last_routed_language: Mutex::new(None),
            last_single_language: Mutex::new(None),
        }
    }

    pub fn preload_for(&self, settings: &AppSettings) -> Result<()> {
        let single_uses_external_detector = settings.transcription_mode
            == TranscriptionMode::Single
            && settings.allowed_languages.len() > 1
            && !uses_native_language_constraint(
                settings.transcription_mode,
                &settings.allowed_languages,
                self.registry.model_architecture(&settings.selected_model),
            );
        let model_id = match settings.transcription_mode {
            TranscriptionMode::Routed if settings.routes.len() > 1 => LANGUAGE_DETECTOR,
            TranscriptionMode::Routed => settings
                .routes
                .first()
                .map(|route| route.model.as_str())
                .context("routing has no routes")?,
            TranscriptionMode::Single if single_uses_external_detector => LANGUAGE_DETECTOR,
            TranscriptionMode::Single => &settings.selected_model,
        };
        self.ensure_loaded(model_id).map(|_| ())
    }

    pub fn transcribe(&self, audio: &[f32], settings: &AppSettings) -> Result<String> {
        let single_uses_native_constraint = uses_native_language_constraint(
            settings.transcription_mode,
            &settings.allowed_languages,
            self.registry.model_architecture(&settings.selected_model),
        );
        let (model_id, source_language) = match settings.transcription_mode {
            TranscriptionMode::Single => (
                settings.selected_model.clone(),
                if single_uses_native_constraint {
                    None
                } else {
                    self.single_language(audio, &settings.allowed_languages)
                },
            ),
            TranscriptionMode::Routed => self.route(audio, settings)?,
        };
        let model = self.ensure_loaded(&model_id)?;
        let caps = model.capabilities();
        let translate = settings.translate_to_english
            && caps.supports_translate
            && source_language.as_deref() != Some("en");
        let mut options = RunOptions {
            task: if translate {
                Task::Translate
            } else {
                Task::Transcribe
            },
            language: source_language,
            target_language: translate.then(|| "en".to_string()),
            ..RunOptions::default()
        };
        if model.arch() == "whisper"
            && (single_uses_native_constraint || !settings.custom_words.is_empty())
        {
            options.family = Some(RunExtension::Whisper(WhisperRunOptions {
                initial_prompt: (!settings.custom_words.is_empty())
                    .then(|| settings.custom_words.join(", ")),
                allowed_languages: single_uses_native_constraint
                    .then(|| settings.allowed_languages.clone()),
                ..WhisperRunOptions::default()
            }));
        }
        // A Session owns the KV cache, encoder output, and scheduler work
        // buffers. Keep the expensive model loaded, but release those buffers
        // as soon as this transcription finishes.
        let transcript = {
            let mut session = model.session()?;
            session.run(audio, &options)
        }
        .map_err(|error| anyhow!("transcription failed: {error}"))?;
        let mut text = transcript.text.trim().to_string();
        if settings.append_trailing_space && !text.is_empty() {
            text.push(' ');
        }
        Ok(text)
    }

    pub fn unload(&self) {
        *self.loaded.lock() = None;
    }

    pub fn loaded_model_id(&self) -> Option<String> {
        self.loaded
            .lock()
            .as_ref()
            .map(|loaded| loaded.model_id.clone())
    }

    fn route(&self, audio: &[f32], settings: &AppSettings) -> Result<(String, Option<String>)> {
        let first = settings.routes.first().context("routing has no routes")?;
        if settings.routes.len() == 1 {
            return Ok((first.model.clone(), Some(first.language.clone())));
        }

        let detected = self.detect_language(audio)?;
        let languages = settings
            .routes
            .iter()
            .map(|route| route.language.clone())
            .collect::<Vec<_>>();
        let language = choose_allowed_language(
            detected.as_deref(),
            self.last_routed_language.lock().as_deref(),
            &languages,
        )
        .context("routing has no languages")?;
        *self.last_routed_language.lock() = Some(language.clone());
        let route = settings
            .routes
            .iter()
            .find(|route| route.language == language)
            .unwrap_or(first);
        Ok((route.model.clone(), Some(route.language.clone())))
    }

    fn single_language(&self, audio: &[f32], allowed: &[String]) -> Option<String> {
        if allowed.is_empty() {
            return None;
        }
        if allowed.len() == 1 {
            return allowed.first().cloned();
        }
        let detected = self.detect_language(audio).ok().flatten();
        let language = choose_allowed_language(
            detected.as_deref(),
            self.last_single_language.lock().as_deref(),
            allowed,
        )?;
        *self.last_single_language.lock() = Some(language.clone());
        Some(language)
    }

    fn detect_language(&self, audio: &[f32]) -> Result<Option<String>> {
        let model = self.ensure_loaded(LANGUAGE_DETECTOR)?;
        let count = audio.len().min(LANGUAGE_DETECT_SAMPLES);
        let detected = {
            let mut session = model.session()?;
            session.run(&audio[..count], &RunOptions::default())
        }
        .ok()
        .and_then(|transcript| transcript.language);
        Ok(detected)
    }

    fn ensure_loaded(&self, model_id: &str) -> Result<Model> {
        let mut loaded = self.loaded.lock();
        if let Some(engine) = loaded.as_ref().filter(|engine| engine.model_id == model_id) {
            return Ok(engine.model.clone());
        }
        let path = self.registry.resolve_path(model_id)?;
        loaded.take();
        let model = Model::load_with(
            &path,
            &ModelOptions {
                backend: Backend::Auto,
                device: None,
            },
        )
        .with_context(|| format!("load {}", path.display()))?;
        log::info!(
            "loaded {} with {} on {:?}",
            model_id,
            model.backend(),
            model.device().ok().map(|device| device.description)
        );
        *loaded = Some(LoadedEngine {
            model_id: model_id.to_string(),
            model: model.clone(),
        });
        Ok(model)
    }
}

fn choose_allowed_language(
    detected: Option<&str>,
    previous: Option<&str>,
    allowed: &[String],
) -> Option<String> {
    detected
        .filter(|detected| allowed.iter().any(|language| language == detected))
        .or_else(|| previous.filter(|previous| allowed.iter().any(|language| language == previous)))
        .map(str::to_string)
        .or_else(|| allowed.first().cloned())
}

fn uses_native_language_constraint(
    mode: TranscriptionMode,
    allowed: &[String],
    architecture: Option<&str>,
) -> bool {
    mode == TranscriptionMode::Single && allowed.len() > 1 && architecture == Some("whisper")
}

#[cfg(test)]
mod tests {
    use super::{choose_allowed_language, uses_native_language_constraint};
    use crate::settings::TranscriptionMode;

    #[test]
    fn routing_prefers_an_allowed_detection() {
        let allowed = vec!["ru".to_string(), "en".to_string(), "fr".to_string()];
        assert_eq!(
            choose_allowed_language(Some("fr"), Some("en"), &allowed).as_deref(),
            Some("fr")
        );
    }

    #[test]
    fn routing_keeps_the_previous_language_for_an_unknown_detection() {
        let allowed = vec!["ru".to_string(), "en".to_string()];
        assert_eq!(
            choose_allowed_language(Some("de"), Some("en"), &allowed).as_deref(),
            Some("en")
        );
    }

    #[test]
    fn single_whisper_uses_its_own_language_detection() {
        let allowed = vec!["ru".to_string(), "en".to_string()];
        assert!(uses_native_language_constraint(
            TranscriptionMode::Single,
            &allowed,
            Some("whisper")
        ));
        assert!(!uses_native_language_constraint(
            TranscriptionMode::Single,
            &allowed,
            Some("parakeet")
        ));
    }
}
