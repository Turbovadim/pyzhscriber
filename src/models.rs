use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize)]
pub struct CatalogFile {
    pub filename: String,
    pub quant: String,
    pub size_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelCapabilities {
    pub streaming: bool,
    pub translate: bool,
    pub lang_detect: bool,
    pub timestamps: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub revision: String,
    pub name: String,
    pub description: String,
    pub architecture: String,
    pub languages: Vec<String>,
    pub capabilities: ModelCapabilities,
    pub speed_score: f32,
    pub accuracy_score: f32,
    pub files: Vec<CatalogFile>,
    pub default_quant: String,
    #[serde(default)]
    pub recommended: bool,
}

#[derive(Deserialize)]
struct CatalogRoot {
    models: Vec<CatalogModel>,
}

#[derive(Clone, Debug)]
pub struct ModelChoice {
    pub id: String,
    pub name: String,
    pub description: String,
    pub architecture: String,
    pub languages: Vec<String>,
    pub quant: String,
    pub size_bytes: u64,
    pub speed_score: f32,
    pub accuracy_score: f32,
    pub installed: bool,
    pub recommended: bool,
    pub streaming: bool,
    pub translate: bool,
}

pub struct ModelRegistry {
    models: Vec<CatalogModel>,
    owned_model_dir: PathBuf,
    hugging_face_hubs: Vec<PathBuf>,
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let root: CatalogRoot =
            serde_json::from_str(include_str!("../resources/model-catalog.json"))
                .context("parse bundled model catalog")?;
        let data_dir = dirs::data_dir()
            .context("application data directory is unavailable")?
            .join(env!("PYZHSCRIBER_BUNDLE_ID"))
            .join("models");
        fs::create_dir_all(&data_dir)?;
        let mut caches = vec![
            dirs::cache_dir()
                .context("cache directory is unavailable")?
                .join("huggingface/hub"),
        ];
        if let Some(home) = dirs::home_dir() {
            let common_cli_cache = home.join(".cache/huggingface/hub");
            if !caches.contains(&common_cli_cache) {
                caches.push(common_cli_cache);
            }
        }
        Ok(Self {
            models: root.models,
            owned_model_dir: data_dir,
            hugging_face_hubs: caches,
        })
    }

    pub fn choices(&self) -> Vec<ModelChoice> {
        let mut choices = Vec::new();
        for model in &self.models {
            for file in &model.files {
                let id = format!("{}/{}", model.id, file.filename);
                choices.push(ModelChoice {
                    installed: self.resolve_path(&id).is_ok(),
                    id,
                    name: model.name.clone(),
                    description: model.description.clone(),
                    architecture: model.architecture.clone(),
                    languages: model.languages.clone(),
                    quant: file.quant.clone(),
                    size_bytes: file.size_bytes,
                    speed_score: model.speed_score / 100.0,
                    accuracy_score: model.accuracy_score / 100.0,
                    recommended: model.recommended && file.quant == model.default_quant,
                    streaming: model.capabilities.streaming,
                    translate: model.capabilities.translate,
                });
            }
        }
        choices.sort_by(|a, b| {
            b.installed
                .cmp(&a.installed)
                .then_with(|| b.recommended.cmp(&a.recommended))
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.size_bytes.cmp(&b.size_bytes))
        });
        choices
    }

    pub fn default_choices(&self) -> Vec<ModelChoice> {
        let mut choices = self
            .choices()
            .into_iter()
            .filter(|choice| {
                self.model_and_file(&choice.id)
                    .is_some_and(|(model, file)| file.quant == model.default_quant)
            })
            .collect::<Vec<_>>();
        choices.sort_by(|a, b| {
            b.installed
                .cmp(&a.installed)
                .then_with(|| b.recommended.cmp(&a.recommended))
                .then_with(|| a.name.cmp(&b.name))
        });
        choices
    }

    pub fn model_name(&self, model_id: &str) -> String {
        self.model_and_file(model_id)
            .map(|(model, _)| model.name.clone())
            .unwrap_or_else(|| model_id.rsplit('/').next().unwrap_or(model_id).to_string())
    }

    pub fn model_architecture(&self, model_id: &str) -> Option<&str> {
        self.model_and_file(model_id)
            .map(|(model, _)| model.architecture.as_str())
    }

    pub fn resolve_path(&self, model_id: &str) -> Result<PathBuf> {
        let (model, file) = self
            .model_and_file(model_id)
            .ok_or_else(|| anyhow!("unknown model: {model_id}"))?;

        let owned = self.owned_model_dir.join(&file.filename);
        if owned.is_file() {
            return Ok(owned);
        }

        let repo_dir = format!("models--{}", model.id.replace('/', "--"));
        for hub in &self.hugging_face_hubs {
            let snapshots = hub.join(&repo_dir).join("snapshots");
            let exact = snapshots.join(&model.revision).join(&file.filename);
            if exact.is_file() {
                return Ok(exact);
            }
            if let Ok(entries) = fs::read_dir(snapshots) {
                for entry in entries.flatten() {
                    let candidate = entry.path().join(&file.filename);
                    if candidate.is_file() {
                        return Ok(candidate);
                    }
                }
            }
        }
        Err(anyhow!("model is not downloaded: {model_id}"))
    }

    pub fn download_spec(&self, model_id: &str) -> Result<ModelDownload> {
        let (model, file) = self
            .model_and_file(model_id)
            .ok_or_else(|| anyhow!("unknown model: {model_id}"))?;
        Ok(ModelDownload {
            url: format!(
                "https://blob.handy.computer/{}/{}/{}",
                model.id, model.revision, file.filename
            ),
            destination: self.owned_model_dir.join(&file.filename),
            expected_sha256: file.sha256.clone(),
            size_bytes: file.size_bytes,
        })
    }

    pub fn download(&self, model_id: &str, mut progress: impl FnMut(f32)) -> Result<PathBuf> {
        let spec = self.download_spec(model_id)?;
        let temporary = spec.destination.with_extension("download");
        let mut response = reqwest::blocking::Client::new()
            .get(&spec.url)
            .send()?
            .error_for_status()?;
        let mut file = fs::File::create(&temporary)?;
        let mut hasher = Sha256::new();
        let mut downloaded = 0_u64;
        let mut buffer = [0_u8; 256 * 1024];
        loop {
            let count = response.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            file.write_all(&buffer[..count])?;
            hasher.update(&buffer[..count]);
            downloaded += count as u64;
            progress((downloaded as f64 / spec.size_bytes as f64).clamp(0.0, 1.0) as f32);
        }
        file.sync_all()?;
        let actual = format!("{:x}", hasher.finalize());
        if actual != spec.expected_sha256 {
            let _ = fs::remove_file(&temporary);
            return Err(anyhow!(
                "download hash mismatch for {model_id}: expected {}, got {actual}",
                spec.expected_sha256
            ));
        }
        fs::rename(&temporary, &spec.destination)?;
        progress(1.0);
        Ok(spec.destination)
    }

    fn model_and_file(&self, model_id: &str) -> Option<(&CatalogModel, &CatalogFile)> {
        self.models.iter().find_map(|model| {
            model.files.iter().find_map(|file| {
                (model_id == format!("{}/{}", model.id, file.filename)).then_some((model, file))
            })
        })
    }
}

#[derive(Clone, Debug)]
pub struct ModelDownload {
    pub url: String,
    pub destination: PathBuf,
    pub expected_sha256: String,
    pub size_bytes: u64,
}

pub fn display_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
}

pub fn is_english_only(choice: &ModelChoice) -> bool {
    choice.languages.as_slice() == ["en"]
}

pub fn supports_english(choice: &ModelChoice) -> bool {
    choice.languages.iter().any(|language| language == "en")
}

pub fn supports_russian(choice: &ModelChoice) -> bool {
    choice.languages.iter().any(|language| language == "ru")
}

pub fn supports_language(choice: &ModelChoice, language: &str) -> bool {
    choice
        .languages
        .iter()
        .any(|supported| supported == language)
}

pub fn language_label(code: &str) -> String {
    let name = match code {
        "ar" => "Arabic",
        "bg" => "Bulgarian",
        "cs" => "Czech",
        "da" => "Danish",
        "de" => "German",
        "el" => "Greek",
        "en" => "English",
        "es" => "Spanish",
        "et" => "Estonian",
        "fa" => "Persian",
        "fi" => "Finnish",
        "fil" | "tl" => "Filipino",
        "fr" => "French",
        "he" => "Hebrew",
        "hi" => "Hindi",
        "hr" => "Croatian",
        "hu" => "Hungarian",
        "id" => "Indonesian",
        "it" => "Italian",
        "ja" => "Japanese",
        "ko" => "Korean",
        "lt" => "Lithuanian",
        "lv" => "Latvian",
        "mk" => "Macedonian",
        "ms" => "Malay",
        "mt" => "Maltese",
        "nb" | "no" => "Norwegian",
        "nl" => "Dutch",
        "pl" => "Polish",
        "pt" => "Portuguese",
        "ro" => "Romanian",
        "ru" => "Russian",
        "sk" => "Slovak",
        "sl" => "Slovenian",
        "sv" => "Swedish",
        "th" => "Thai",
        "tr" => "Turkish",
        "uk" => "Ukrainian",
        "vi" => "Vietnamese",
        "yue" => "Cantonese",
        "zh" => "Chinese",
        _ => return code.to_uppercase(),
    };
    format!("{name} · {}", code.to_uppercase())
}

pub fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_catalog_contains_routing_models() {
        let registry = ModelRegistry::new().unwrap();
        let choices = registry.default_choices();
        assert!(choices.iter().any(is_english_only));
        assert!(choices.iter().any(supports_english));
        assert!(choices.iter().any(supports_russian));
    }
}
