use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Static hosts tried in order; a file lives at `{mirror}/{repo}/{revision}/{filename}`.
/// The catalog package no longer lists them, so they are pinned here as in Handy.
const MIRRORS: &[&str] = &["https://blob.handy.computer"];

/// List order and the "Recommended" badge, keyed by catalog id. The package
/// carries no ranking, so this is Handy's editorial order, kept from the
/// previous catalog format.
const RANKING: &[(&str, u32, bool)] = &[
    ("parakeet-unified-en-0.6b", 1, true),
    ("nemotron-3.5-asr-streaming-0.6b", 2, true),
    ("canary-180m-flash", 3, true),
    ("cohere-transcribe-03-2026", 4, true),
    ("whisper-medium", 5, true),
    ("Voxtral-Mini-4B-Realtime-2602", 6, false),
    ("parakeet-tdt-0.6b-v3", 7, false),
    ("parakeet-tdt-0.6b-v2", 8, false),
    ("Qwen3-ASR-0.6B", 9, false),
    ("Fun-ASR-MLT-Nano-2512", 10, false),
];

/// `resources/model-catalog.json` is a verbatim copy of the `catalog.json` in
/// the `@handy-computer/models` npm package. Only the fields we use are
/// declared; serde ignores the rest.
#[derive(Deserialize)]
struct PackageCatalog {
    models: Vec<PackageModel>,
    /// Benchmark results keyed by catalog id.
    #[serde(default)]
    results: HashMap<String, PackageResults>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PackageModel {
    id: String,
    name: String,
    /// Editorial copy; empty for models that are not curated.
    description: String,
    /// Hugging Face repo, e.g. `handy-computer/whisper-small-gguf`.
    repository: String,
    revision: String,
    /// The loader architecture (`general.architecture`).
    family: String,
    languages: Vec<String>,
    capabilities: ModelCapabilities,
    /// Key of the default file in `files`.
    quant: String,
    /// `asr`, `langid`, `vad` or `diarize`; only `asr` models transcribe.
    role: String,
    files: HashMap<String, PackageFile>,
}

#[derive(Deserialize)]
struct PackageFile {
    filename: String,
    size: u64,
    sha256: String,
}

#[derive(Deserialize, Default)]
struct PackageResults {
    /// Error percent per base language code, measured at the default quant.
    #[serde(default)]
    accuracy: HashMap<String, AccuracyCell>,
    #[serde(default)]
    performance: Performance,
}

#[derive(Deserialize)]
struct AccuracyCell {
    error: f64,
}

#[derive(Deserialize, Default)]
struct Performance {
    /// Real-time factor on the benchmark GPU, then on its CPU.
    accelerated: Option<f64>,
    cpu: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelCapabilities {
    streaming: bool,
    translate: bool,
    language_detection: bool,
    #[serde(default)]
    timestamps: bool,
}

#[derive(Clone, Debug)]
struct CatalogFile {
    filename: String,
    quant: String,
    size_bytes: u64,
    sha256: String,
}

/// A transcription model as the registry uses it, normalised from the package.
#[derive(Clone, Debug)]
struct CatalogModel {
    /// Hugging Face repo id; with a filename it forms the ids settings store.
    id: String,
    revision: String,
    name: String,
    description: String,
    architecture: String,
    languages: Vec<String>,
    capabilities: ModelCapabilities,
    /// 0 to 100, from the package's benchmarks.
    speed_score: f32,
    accuracy_score: f32,
    /// Largest first, so the first installed alternate is the best one.
    files: Vec<CatalogFile>,
    default_quant: String,
    recommended: bool,
    recommended_rank: Option<u32>,
}

/// Parses the bundled package catalog, keeping only models that transcribe.
fn load_catalog(json: &str) -> Result<Vec<CatalogModel>> {
    let mut catalog: PackageCatalog =
        serde_json::from_str(json).context("parse bundled model catalog")?;
    catalog
        .models
        .into_iter()
        .filter(|model| model.role == "asr")
        .map(|model| {
            anyhow::ensure!(
                model.files.contains_key(&model.quant),
                "catalog model {} has no {} file",
                model.id,
                model.quant
            );
            let results = catalog.results.remove(&model.id);
            let ranking = RANKING.iter().find(|(id, _, _)| *id == model.id);
            let mut files = model
                .files
                .into_iter()
                .map(|(quant, file)| CatalogFile {
                    filename: file.filename,
                    quant,
                    size_bytes: file.size,
                    sha256: file.sha256,
                })
                .collect::<Vec<_>>();
            files.sort_by_key(|file| std::cmp::Reverse(file.size_bytes));
            let description = if model.description.is_empty() {
                generated_description(&model.languages, &model.capabilities)
            } else {
                model.description
            };
            Ok(CatalogModel {
                speed_score: speed_score(results.as_ref()),
                accuracy_score: accuracy_score(&model.languages, results.as_ref()),
                id: model.repository,
                revision: model.revision,
                name: model.name,
                description,
                architecture: model.family,
                languages: model.languages,
                capabilities: model.capabilities,
                files,
                default_quant: model.quant,
                recommended: ranking.is_some_and(|(_, _, recommended)| *recommended),
                recommended_rank: ranking.map(|(_, rank, _)| *rank),
            })
        })
        .collect()
}

/// speed = 100·(1 − e^(−rtf/8)); a model too heavy to benchmark counts as slow.
fn speed_score(results: Option<&PackageResults>) -> f32 {
    let rtf =
        results.and_then(|results| results.performance.accelerated.or(results.performance.cpu));
    match rtf {
        Some(rtf) if rtf > 0.0 => (100.0 * (1.0 - (-rtf / 8.0).exp())).round() as f32,
        _ => 10.0,
    }
}

/// accuracy = 100·e^(−error/15), on English when measured, else on the
/// model's first language.
fn accuracy_score(languages: &[String], results: Option<&PackageResults>) -> f32 {
    let Some(results) = results else {
        return 0.0;
    };
    std::iter::once("en".to_string())
        .chain(languages.first().map(|language| base_language(language)))
        .find_map(|language| results.accuracy.get(&language))
        .map_or(0.0, |cell| {
            (100.0 * (-cell.error / 15.0).exp()).round() as f32
        })
}

fn base_language(language: &str) -> String {
    language
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .to_ascii_lowercase()
}

/// A factual one-liner for models the package has no editorial copy for.
fn generated_description(languages: &[String], capabilities: &ModelCapabilities) -> String {
    let features = [
        (capabilities.translate, "translation"),
        (capabilities.language_detection, "auto language detection"),
        (capabilities.streaming, "streaming"),
        (capabilities.timestamps, "timestamps"),
    ]
    .into_iter()
    .filter_map(|(present, feature)| present.then_some(feature))
    .collect::<Vec<_>>();
    let base = match languages {
        [language] => format!(
            "{} speech-to-text",
            language_label(language)
                .split(" · ")
                .next()
                .unwrap_or(language)
        ),
        languages => format!("{}-language speech-to-text", languages.len()),
    };
    if features.is_empty() {
        format!("{base}.")
    } else {
        format!("{base} with {}.", features.join(", "))
    }
}

/// One entry per catalog model. `id` always names the default-quant file,
/// which is what settings store; `quant` and `size_bytes` describe the file
/// that is actually used, i.e. the installed quant when one is on disk.
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
    /// The installed file lives in our own models folder rather than in a
    /// shared Hugging Face cache, so deleting it cannot break another tool.
    pub removable: bool,
    /// Bytes already fetched by an interrupted download, resumed on retry.
    pub partial_bytes: u64,
    pub recommended: bool,
    pub streaming: bool,
    pub translate: bool,
}

pub struct ModelRegistry {
    models: Vec<CatalogModel>,
    owned_model_dir: PathBuf,
    hugging_face_hubs: Vec<PathBuf>,
}

/// Why a download stopped without producing a file.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("download cancelled")]
    Cancelled,
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

impl ModelRegistry {
    pub fn new() -> Result<Self> {
        let models = load_catalog(include_str!("../resources/model-catalog.json"))?;
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
            models,
            owned_model_dir: data_dir,
            hugging_face_hubs: caches,
        })
    }

    /// Every catalog model, installed first, then by editorial rank and name.
    /// Cheap enough to call on every page switch: a few stats per model.
    pub fn choices(&self) -> Vec<ModelChoice> {
        let mut choices = self
            .models
            .iter()
            .map(|model| {
                let default = model.default_file();
                let installed = self.installed_file(model);
                let file = installed.as_ref().map_or(default, |(file, _)| file);
                ModelChoice {
                    id: model.file_id(default),
                    name: model.name.clone(),
                    description: model.description.clone(),
                    architecture: model.architecture.clone(),
                    languages: model.languages.clone(),
                    quant: file.quant.clone(),
                    size_bytes: file.size_bytes,
                    speed_score: model.speed_score / 100.0,
                    accuracy_score: model.accuracy_score / 100.0,
                    installed: installed.is_some(),
                    removable: installed
                        .as_ref()
                        .is_some_and(|(_, path)| path.starts_with(&self.owned_model_dir)),
                    partial_bytes: fs::metadata(self.partial_path(default))
                        .map_or(0, |metadata| metadata.len()),
                    recommended: model.recommended,
                    streaming: model.capabilities.streaming,
                    translate: model.capabilities.translate,
                }
            })
            .collect::<Vec<_>>();
        let rank = |id: &str| {
            self.model(id)
                .and_then(|model| model.recommended_rank)
                .unwrap_or(u32::MAX)
        };
        choices.sort_by(|a, b| {
            b.installed
                .cmp(&a.installed)
                .then_with(|| rank(&a.id).cmp(&rank(&b.id)))
                .then_with(|| b.recommended.cmp(&a.recommended))
                .then_with(|| a.name.cmp(&b.name))
        });
        choices
    }

    /// Maps any quant's id to its model's default-file id, the form the rest
    /// of the app stores. Unknown ids pass through untouched.
    pub fn canonical_id(&self, model_id: &str) -> String {
        self.model(model_id).map_or_else(
            || model_id.to_string(),
            |model| model.file_id(model.default_file()),
        )
    }

    pub fn model_name(&self, model_id: &str) -> String {
        self.model(model_id)
            .map(|model| model.name.clone())
            .unwrap_or_else(|| model_id.rsplit('/').next().unwrap_or(model_id).to_string())
    }

    pub fn model_architecture(&self, model_id: &str) -> Option<&str> {
        self.model(model_id)
            .map(|model| model.architecture.as_str())
    }

    /// The exact file `model_id` names when it is on disk, otherwise any other
    /// installed quant of the same model.
    pub fn resolve_path(&self, model_id: &str) -> Result<PathBuf> {
        let model = self
            .model(model_id)
            .ok_or_else(|| anyhow!("unknown model: {model_id}"))?;
        let exact = model
            .files
            .iter()
            .find(|file| model.file_id(file) == model_id)
            .and_then(|file| self.locate(model, file));
        exact
            .or_else(|| self.installed_file(model).map(|(_, path)| path))
            .ok_or_else(|| anyhow!("model is not downloaded: {}", model.name))
    }

    /// Downloads the default quant into our models folder, resuming any
    /// partial file and trying each mirror in turn. The catalog hash compiled
    /// into the binary is the trust anchor, so mirrors are plain bit pipes.
    pub fn download(
        &self,
        model_id: &str,
        cancel: &AtomicBool,
        mut progress: impl FnMut(f32),
    ) -> Result<PathBuf, DownloadError> {
        let model = self
            .model(model_id)
            .ok_or_else(|| anyhow!("unknown model: {model_id}"))?;
        let file = model.default_file();
        let destination = self.owned_model_dir.join(&file.filename);
        let partial = self.partial_path(file);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).context("create model folder")?;
        }
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            // Per read in the blocking client, so this catches a stalled stream.
            .timeout(Duration::from_secs(30))
            .build()
            .context("build HTTP client")?;

        let mut last_error = anyhow!("no download mirrors are configured");
        for mirror in MIRRORS {
            let url = format!(
                "{}/{}/{}/{}",
                mirror.trim_end_matches('/'),
                model.id,
                model.revision,
                file.filename
            );
            match fetch_resumable(
                &client,
                &url,
                &partial,
                file.size_bytes,
                cancel,
                &mut progress,
            ) {
                Ok(()) => {
                    let actual = sha256_file(&partial)?;
                    if actual != file.sha256 {
                        let _ = fs::remove_file(&partial);
                        return Err(anyhow!(
                            "download of {} is corrupt (hash mismatch); try again",
                            model.name
                        )
                        .into());
                    }
                    fs::rename(&partial, &destination).context("finish download")?;
                    progress(1.0);
                    return Ok(destination);
                }
                Err(DownloadError::Cancelled) => return Err(DownloadError::Cancelled),
                Err(DownloadError::Failed(error)) => {
                    log::warn!("download from {url} failed: {error:#}");
                    last_error = error;
                }
            }
        }
        Err(last_error
            .context(format!("download {}", model.name))
            .into())
    }

    /// Deletes every quant and partial download of a model from our own models
    /// folder. Files in shared Hugging Face caches are left alone.
    pub fn remove(&self, model_id: &str) -> Result<()> {
        let model = self
            .model(model_id)
            .ok_or_else(|| anyhow!("unknown model: {model_id}"))?;
        for file in &model.files {
            for path in [
                self.owned_model_dir.join(&file.filename),
                self.partial_path(file),
            ] {
                match fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(error).with_context(|| format!("delete {}", path.display()));
                    }
                }
            }
        }
        Ok(())
    }

    /// The model owning `model_id`, matched against every listed quant so ids
    /// saved for a non-default file (e.g. imported from Handy) still resolve.
    fn model(&self, model_id: &str) -> Option<&CatalogModel> {
        self.models.iter().find(|model| {
            model_id
                .strip_prefix(model.id.as_str())
                .and_then(|rest| rest.strip_prefix('/'))
                .is_some_and(|filename| model.files.iter().any(|file| file.filename == filename))
        })
    }

    /// The installed quant to use: the default when present, otherwise the
    /// first other catalog quant found on disk.
    fn installed_file<'a>(&self, model: &'a CatalogModel) -> Option<(&'a CatalogFile, PathBuf)> {
        let default = model.default_file();
        std::iter::once(default)
            .chain(
                model
                    .files
                    .iter()
                    .filter(|file| !std::ptr::eq(*file, default)),
            )
            .find_map(|file| self.locate(model, file).map(|path| (file, path)))
    }

    fn locate(&self, model: &CatalogModel, file: &CatalogFile) -> Option<PathBuf> {
        let owned = self.owned_model_dir.join(&file.filename);
        if owned.is_file() {
            return Some(owned);
        }
        let repo_dir = format!("models--{}", model.id.replace('/', "--"));
        self.hugging_face_hubs.iter().find_map(|hub| {
            let snapshots = hub.join(&repo_dir).join("snapshots");
            let exact = snapshots.join(&model.revision).join(&file.filename);
            if exact.is_file() {
                return Some(exact);
            }
            fs::read_dir(snapshots).ok()?.flatten().find_map(|entry| {
                let candidate = entry.path().join(&file.filename);
                candidate.is_file().then_some(candidate)
            })
        })
    }

    fn partial_path(&self, file: &CatalogFile) -> PathBuf {
        self.owned_model_dir
            .join(format!("{}.partial", file.filename))
    }
}

impl CatalogModel {
    /// `load_catalog` guarantees the default quant has a file.
    fn default_file(&self) -> &CatalogFile {
        self.files
            .iter()
            .find(|file| file.quant == self.default_quant)
            .expect("catalog models include their default quant")
    }

    fn file_id(&self, file: &CatalogFile) -> String {
        format!("{}/{}", self.id, file.filename)
    }
}

/// Streams `url` into `partial`, continuing from its current length. Leaves
/// the partial in place on failure or cancel so the next attempt resumes.
fn fetch_resumable(
    client: &reqwest::blocking::Client,
    url: &str,
    partial: &Path,
    size: u64,
    cancel: &AtomicBool,
    progress: &mut impl FnMut(f32),
) -> Result<(), DownloadError> {
    let mut offset = fs::metadata(partial).map_or(0, |metadata| metadata.len());
    if offset > size {
        fs::remove_file(partial).context("discard oversized partial download")?;
        offset = 0;
    }
    if offset == size {
        return Ok(());
    }
    let mut request = client.get(url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let mut response = request
        .send()
        .context("connect")?
        .error_for_status()
        .context("request")?;
    // A server that ignores the range resends the whole file from the start.
    let resumed = response.status() == reqwest::StatusCode::PARTIAL_CONTENT
        && response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(content_range_start)
            == Some(offset);
    if !resumed {
        offset = 0;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resumed)
        .truncate(!resumed)
        .open(partial)
        .context("open partial download")?;
    let mut downloaded = offset;
    let mut buffer = vec![0_u8; 256 * 1024];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let count = response.read(&mut buffer).context("read")?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count]).context("write")?;
        downloaded += count as u64;
        progress((downloaded as f64 / size as f64).clamp(0.0, 0.999) as f32);
    }
    file.sync_all().context("flush download")?;
    if downloaded != size {
        return Err(anyhow!("connection closed at {downloaded} of {size} bytes").into());
    }
    Ok(())
}

/// Start offset of a `Content-Range: bytes <start>-<end>/<total>` header.
fn content_range_start(value: &str) -> Option<u64> {
    value
        .strip_prefix("bytes ")?
        .split('-')
        .next()?
        .parse()
        .ok()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).context("open download for verification")?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).context("hash download")?;
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn display_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else {
        format!("{} MB", bytes / 1_000_000)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_in(dir: &Path) -> ModelRegistry {
        let mut registry = ModelRegistry::new().unwrap();
        registry.owned_model_dir = dir.to_path_buf();
        registry.hugging_face_hubs.clear();
        registry
    }

    #[test]
    fn bundled_catalog_lists_each_model_once_with_routing_coverage() {
        let registry = ModelRegistry::new().unwrap();
        let choices = registry.choices();
        let mut ids = choices.iter().map(|choice| &choice.id).collect::<Vec<_>>();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), registry.models.len());
        assert!(choices.iter().any(supports_english));
        assert!(choices.iter().any(supports_russian));
    }

    #[test]
    fn catalog_covers_default_models_and_ranking() {
        let registry = ModelRegistry::new().unwrap();
        let settings = crate::settings::AppSettings::default();
        let defaults = settings
            .routes
            .iter()
            .map(|route| route.model.as_str())
            .chain([
                settings.selected_model.as_str(),
                crate::transcription::LANGUAGE_DETECTOR,
            ]);
        for id in defaults {
            assert!(
                registry.model(id).is_some(),
                "{id} is missing from the catalog"
            );
        }
        let ranked = registry
            .models
            .iter()
            .filter(|model| model.recommended_rank.is_some())
            .count();
        assert_eq!(
            ranked,
            RANKING.len(),
            "a ranked model is missing from the catalog"
        );
    }

    #[test]
    fn alternate_quant_on_disk_counts_as_installed() {
        let dir = std::env::temp_dir().join(format!("pyzhscriber-models-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let registry = registry_in(&dir);
        let model = &registry.models[0];
        let alternate = model
            .files
            .iter()
            .find(|file| file.quant != model.default_quant)
            .unwrap();
        fs::write(dir.join(&alternate.filename), b"").unwrap();

        let default_id = model.file_id(model.default_file());
        let choice = registry
            .choices()
            .into_iter()
            .find(|choice| choice.id == default_id)
            .unwrap();
        assert!(choice.installed && choice.removable);
        assert_eq!(choice.quant, alternate.quant);
        assert_eq!(
            registry.resolve_path(&default_id).unwrap(),
            dir.join(&alternate.filename)
        );
        assert_eq!(registry.canonical_id(&model.file_id(alternate)), default_id);

        registry.remove(&default_id).unwrap();
        assert!(registry.resolve_path(&default_id).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[ignore = "downloads Whisper Tiny (46 MB) from the mirror"]
    fn interrupted_download_resumes_and_verifies() {
        let dir = std::env::temp_dir().join(format!("pyzhscriber-resume-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let registry = registry_in(&dir);
        let model = registry
            .models
            .iter()
            .find(|model| model.id == "handy-computer/whisper-tiny-gguf")
            .unwrap();
        let id = model.file_id(model.default_file());

        let cancel = AtomicBool::new(false);
        let stopped = registry.download(&id, &cancel, |progress| {
            if progress > 0.3 {
                cancel.store(true, Ordering::Relaxed);
            }
        });
        assert!(matches!(stopped, Err(DownloadError::Cancelled)));
        let partial = registry.partial_path(model.default_file());
        let kept = fs::metadata(&partial).unwrap().len();
        assert!(kept > 0);

        let mut first = None;
        let path = registry
            .download(&id, &AtomicBool::new(false), |progress| {
                first.get_or_insert(progress);
            })
            .unwrap();
        assert!(first.unwrap() >= kept as f32 / model.default_file().size_bytes as f32 - 0.01);
        assert_eq!(sha256_file(&path).unwrap(), model.default_file().sha256);
        assert!(!partial.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn content_range_start_reads_the_offset() {
        assert_eq!(content_range_start("bytes 1024-2047/4096"), Some(1024));
        assert_eq!(content_range_start("bytes */4096"), None);
    }
}
