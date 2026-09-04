use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_SINGLE_MODEL: &str =
    "handy-computer/whisper-large-v3-turbo-gguf/whisper-large-v3-turbo-Q8_0.gguf";
pub const DEFAULT_RUSSIAN_MODEL: &str =
    "handy-computer/gigaam-v3-e2e-ctc-gguf/gigaam-v3-e2e-ctc-Q8_0.gguf";
pub const DEFAULT_ENGLISH_MODEL: &str =
    "handy-computer/parakeet-unified-en-0.6b-gguf/parakeet-unified-en-0.6b-Q8_0.gguf";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionMode {
    #[default]
    Routed,
    Single,
}

impl TranscriptionMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Routed => "Routed",
            Self::Single => "Single model",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteSetting {
    pub language: String,
    pub model: String,
}

impl RouteSetting {
    pub fn new(language: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            model: model.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PasteMethod {
    CommandV,
    ClipboardOnly,
    #[default]
    DirectTyping,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct AppSettings {
    pub transcription_mode: TranscriptionMode,
    pub selected_model: String,
    pub routes: Vec<RouteSetting>,
    /// Empty means the selected model may auto-detect any language it supports.
    pub allowed_languages: Vec<String>,
    pub shortcut: String,
    pub push_to_talk: bool,
    pub selected_microphone: Option<String>,
    pub translate_to_english: bool,
    pub custom_words: Vec<String>,
    pub filler_word_removal_enabled: bool,
    pub append_trailing_space: bool,
    pub auto_submit: bool,
    pub paste_method: PasteMethod,
    pub preserve_clipboard: bool,
    pub paste_delay_ms: u64,
    pub audio_feedback: bool,
    pub vad_enabled: bool,
    pub show_overlay: bool,
    pub bubbly_overlay: bool,
    pub start_hidden: bool,
    pub show_tray_icon: bool,
    pub autostart_enabled: bool,
    pub history_limit: usize,
    pub recording_retention_days: Option<u32>,
    pub model_unload_minutes: Option<u32>,
    pub post_process_enabled: bool,
    pub post_process_base_url: String,
    pub post_process_model: String,
    pub post_process_api_key: String,
    pub post_process_prompt: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            transcription_mode: TranscriptionMode::Routed,
            selected_model: DEFAULT_SINGLE_MODEL.into(),
            routes: default_routes(),
            allowed_languages: Vec::new(),
            shortcut: "command_right".into(),
            push_to_talk: true,
            selected_microphone: None,
            translate_to_english: false,
            custom_words: Vec::new(),
            filler_word_removal_enabled: true,
            append_trailing_space: false,
            auto_submit: false,
            paste_method: PasteMethod::DirectTyping,
            preserve_clipboard: true,
            paste_delay_ms: 60,
            audio_feedback: false,
            vad_enabled: true,
            show_overlay: true,
            bubbly_overlay: false,
            start_hidden: false,
            show_tray_icon: true,
            autostart_enabled: false,
            history_limit: 50,
            recording_retention_days: None,
            model_unload_minutes: Some(5),
            post_process_enabled: false,
            post_process_base_url: "https://api.openai.com/v1".into(),
            post_process_model: String::new(),
            post_process_api_key: String::new(),
            post_process_prompt: "Clean the transcript without changing its meaning. Return only the cleaned text.\n\n${output}".into(),
        }
    }
}

pub struct SettingsStore {
    path: PathBuf,
}

impl SettingsStore {
    pub fn new() -> Result<Self> {
        let base = dirs::data_dir().context("macOS application data directory is unavailable")?;
        let directory = base.join(env!("PYZHSCRIBER_BUNDLE_ID"));
        fs::create_dir_all(&directory)
            .with_context(|| format!("create {}", directory.display()))?;
        Ok(Self {
            path: directory.join("settings.json"),
        })
    }

    pub fn load(&self) -> Result<AppSettings> {
        if self.path.exists() {
            let bytes =
                fs::read(&self.path).with_context(|| format!("read {}", self.path.display()))?;
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse {}", self.path.display()))?;
            let (settings, migrated) = deserialize_settings(value)
                .with_context(|| format!("parse {}", self.path.display()))?;
            if migrated {
                self.save(&settings)?;
            }
            return Ok(settings);
        }

        let imported = import_handy_settings().unwrap_or_default();
        self.save(&imported)?;
        Ok(imported)
    }

    pub fn save(&self, settings: &AppSettings) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(settings)?;
        let temporary = self.path.with_extension("json.tmp");
        fs::write(&temporary, bytes).with_context(|| format!("write {}", temporary.display()))?;
        fs::rename(&temporary, &self.path)
            .with_context(|| format!("replace {}", self.path.display()))?;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn import_handy_settings() -> Result<AppSettings> {
    let path = dirs::data_dir()
        .context("application data directory is unavailable")?
        .join("com.pais.handy/settings_store.json");
    let root: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let source = root
        .get("settings")
        .context("Handy settings key is missing")?;
    let mut settings = AppSettings::default();

    if let Some(value) = source.get("selected_model").and_then(|v| v.as_str()) {
        settings.selected_model = value.into();
    }
    if let Some(value) = source.get("transcription_mode").and_then(|v| v.as_str()) {
        settings.transcription_mode = if value == "single" {
            TranscriptionMode::Single
        } else {
            TranscriptionMode::Routed
        };
    }
    if let Some(binding) = source.pointer("/bindings/transcribe") {
        let mut russian_model = DEFAULT_RUSSIAN_MODEL.to_string();
        let mut english_model = DEFAULT_ENGLISH_MODEL.to_string();
        if let Some(value) = binding.get("model_id").and_then(|v| v.as_str()) {
            russian_model = value.into();
        }
        if let Some(value) = binding.get("english_model_id").and_then(|v| v.as_str()) {
            english_model = value.into();
        }
        settings.routes = vec![
            RouteSetting::new("ru", russian_model),
            RouteSetting::new("en", english_model),
        ];
        if let Some(value) = binding.get("current_binding").and_then(|v| v.as_str()) {
            settings.shortcut = value.into();
        }
    }
    settings.push_to_talk = bool_at(source, "push_to_talk", settings.push_to_talk);
    settings.translate_to_english = bool_at(source, "translate_to_english", false);
    settings.filler_word_removal_enabled = bool_at(source, "filler_word_removal_enabled", true);
    settings.append_trailing_space = bool_at(source, "append_trailing_space", false);
    settings.auto_submit = bool_at(source, "auto_submit", false);
    settings.audio_feedback = bool_at(source, "audio_feedback", false);
    settings.vad_enabled = bool_at(source, "vad_enabled", true);
    settings.start_hidden = bool_at(source, "start_hidden", false);
    settings.show_tray_icon = bool_at(source, "show_tray_icon", true);
    settings.autostart_enabled = bool_at(source, "autostart_enabled", false);
    settings.post_process_enabled = bool_at(source, "post_process_enabled", false);
    settings.selected_microphone = source
        .get("selected_microphone")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    settings.custom_words = source
        .get("custom_words")
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Ok(settings)
}

fn default_routes() -> Vec<RouteSetting> {
    vec![
        RouteSetting::new("ru", DEFAULT_RUSSIAN_MODEL),
        RouteSetting::new("en", DEFAULT_ENGLISH_MODEL),
    ]
}

/// Upgrades the two fixed routing slots and the old single-language field on read.
fn deserialize_settings(value: serde_json::Value) -> Result<(AppSettings, bool)> {
    let has_routes = value.get("routes").is_some();
    let has_allowed_languages = value.get("allowed_languages").is_some();
    let mut settings: AppSettings = serde_json::from_value(value.clone())?;
    let mut migrated = false;

    if !has_routes {
        let russian_model = value
            .get("russian_model")
            .and_then(|value| value.as_str())
            .unwrap_or(DEFAULT_RUSSIAN_MODEL);
        let english_model = value
            .get("english_model")
            .and_then(|value| value.as_str())
            .unwrap_or(DEFAULT_ENGLISH_MODEL);
        settings.routes = vec![
            RouteSetting::new("ru", russian_model),
            RouteSetting::new("en", english_model),
        ];
        migrated = true;
    }

    if !has_allowed_languages {
        if let Some(language) = value
            .get("selected_language")
            .and_then(|value| value.as_str())
            .filter(|language| *language != "auto")
        {
            settings.allowed_languages = vec![language.to_string()];
        }
        migrated = true;
    }

    if settings.routes.is_empty() {
        settings.routes = default_routes();
        migrated = true;
    }
    Ok((settings, migrated))
}

fn bool_at(source: &serde_json::Value, key: &str, default: bool) -> bool {
    source.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_keep_routed_and_single_models_independent() {
        let settings = AppSettings::default();
        assert_eq!(settings.transcription_mode, TranscriptionMode::Routed);
        assert_ne!(settings.selected_model, settings.routes[0].model);
        assert_ne!(settings.routes[0].model, settings.routes[1].model);
    }

    #[test]
    fn direct_typing_is_the_default_output_method() {
        assert_eq!(
            AppSettings::default().paste_method,
            PasteMethod::DirectTyping
        );
    }

    #[test]
    fn old_fixed_routes_are_migrated() {
        let value = serde_json::json!({
            "russian_model": "custom/ru.gguf",
            "english_model": "custom/en.gguf",
            "selected_language": "fr"
        });
        let (settings, migrated) = deserialize_settings(value).unwrap();
        assert!(migrated);
        assert_eq!(
            settings.routes,
            vec![
                RouteSetting::new("ru", "custom/ru.gguf"),
                RouteSetting::new("en", "custom/en.gguf")
            ]
        );
        assert_eq!(settings.allowed_languages, ["fr"]);
    }
}
