//! Live sessions for streaming models. Audio goes out over a WebSocket while
//! it is recorded, so once recording stops only the last moment of speech is
//! left to transcribe.

use super::{Endpoint, keywords, language_hints};
use crate::audio::{AudioTap, Resampler};
use crate::settings::{AppSettings, CloudProvider};
use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::sync::mpsc;
use std::time::Duration;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderValue, header::AUTHORIZATION};
use tokio_tungstenite::tungstenite::{self, Message};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long `finish` waits for the transcript, matching the upload timeout.
const FINISH_TIMEOUT: Duration = Duration::from_secs(60);
/// The rate the recorder taps audio at.
const INPUT_RATE: u32 = 16_000;

/// The wire dialect of a streaming model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Protocol {
    /// OpenAI's Realtime API in transcription mode.
    OpenAi,
    /// Mistral's realtime Voxtral endpoint.
    Mistral,
}

impl Protocol {
    pub(super) fn of(provider: CloudProvider, model: &str) -> Option<Self> {
        match provider {
            CloudProvider::OpenAi
                if ["gpt-live-transcribe", "gpt-realtime-whisper"]
                    .iter()
                    .any(|prefix| model.starts_with(prefix)) =>
            {
                Some(Self::OpenAi)
            }
            CloudProvider::Mistral
                if model.starts_with("voxtral") && model.contains("realtime") =>
            {
                Some(Self::Mistral)
            }
            _ => None,
        }
    }

    /// OpenAI only takes 24 kHz PCM; Mistral takes the recorder's 16 kHz.
    fn sample_rate(self) -> u32 {
        match self {
            Self::OpenAi => 24_000,
            Self::Mistral => INPUT_RATE,
        }
    }

    fn url(self, base_url: &str, model: &str) -> Result<String> {
        let base = base_url.replacen("https://", "wss://", 1);
        let url = match self {
            Self::OpenAi => reqwest::Url::parse_with_params(
                &format!("{base}/realtime"),
                [("intent", "transcription")],
            ),
            Self::Mistral => reqwest::Url::parse_with_params(
                &format!("{base}/audio/transcriptions/realtime"),
                [("model", model)],
            ),
        };
        Ok(url?.into())
    }

    /// Session settings, sent once before any audio. As with uploads, a
    /// single-language field is only set when exactly one language is expected.
    fn setup(self, model: &str, languages: &[String], words: &[String]) -> Value {
        match self {
            Self::OpenAi => {
                let mut transcription = json!({ "model": model });
                if model.starts_with("gpt-live-transcribe") {
                    if !languages.is_empty() {
                        transcription["languages"] = json!(languages);
                    }
                    let keywords = keywords(words).collect::<Vec<_>>();
                    if !keywords.is_empty() {
                        transcription["keywords"] = json!(keywords);
                    }
                } else {
                    if let [language] = languages {
                        transcription["language"] = json!(language);
                    }
                    if !words.is_empty() {
                        transcription["prompt"] = json!(words.join(", "));
                    }
                }
                json!({
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": { "type": "audio/pcm", "rate": self.sample_rate() },
                                "transcription": transcription,
                                // Recording ends the turn, so the server never splits it.
                                "turn_detection": null,
                            },
                        },
                    },
                })
            }
            Self::Mistral => {
                let mut session = json!({
                    "audio_format": { "encoding": "pcm_s16le", "sample_rate": self.sample_rate() },
                    "target_streaming_delay_ms": 480,
                    "context_bias": words,
                });
                if let [language] = languages {
                    session["language"] = json!(language);
                }
                json!({ "type": "session.update", "session": session })
            }
        }
    }

    /// Carries 16-bit PCM at `sample_rate`.
    fn append(self, samples: &[f32]) -> Value {
        let bytes = samples
            .iter()
            .flat_map(|sample| ((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())
            .collect::<Vec<_>>();
        let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
        match self {
            Self::OpenAi => json!({ "type": "input_audio_buffer.append", "audio": audio }),
            Self::Mistral => json!({ "type": "input_audio.append", "audio": audio }),
        }
    }

    fn end(self) -> Value {
        match self {
            Self::OpenAi => json!({ "type": "input_audio_buffer.commit" }),
            Self::Mistral => json!({ "type": "input_audio.end" }),
        }
    }

    /// The transcript once a server event carries the final one, `None` for
    /// any other event.
    fn read(self, event: &Value) -> Result<Option<String>> {
        let text = match (self, event["type"].as_str().unwrap_or_default()) {
            (_, "error") | (Self::OpenAi, "conversation.item.input_audio_transcription.failed") => {
                bail!(
                    "{}",
                    event["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown error")
                )
            }
            (Self::OpenAi, "conversation.item.input_audio_transcription.completed") => {
                &event["transcript"]
            }
            (Self::Mistral, "transcription.done") => &event["text"],
            _ => return Ok(None),
        };
        text.as_str()
            .map(|text| Some(text.trim().to_string()))
            .context("the final event has no transcript")
    }
}

/// A streaming transcription in progress. Dropping it before `finish`
/// returns abandons the session.
pub struct LiveSession {
    label: &'static str,
    result: mpsc::Receiver<Result<String>>,
    /// Its receiver sees the channel close once the session is dropped.
    _alive: async_channel::Sender<()>,
}

impl LiveSession {
    /// Connects in the background and returns the tap to feed 16 kHz audio
    /// into; audio queues until the connection is up. Dropping the tap ends
    /// the input. Every problem, missing settings included, surfaces from
    /// `finish`.
    pub fn start(settings: &AppSettings) -> (Self, AudioTap) {
        let (audio_tx, audio_rx) = async_channel::unbounded::<Vec<f32>>();
        let (alive_tx, alive_rx) = async_channel::bounded::<()>(1);
        let (result_tx, result) = mpsc::channel();
        let label = settings.cloud_provider.label();
        let settings = settings.clone();
        std::thread::spawn(move || {
            let outcome = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .context("could not start streaming")
                .and_then(|runtime| {
                    runtime.block_on(async {
                        tokio::select! {
                            outcome = stream(&settings, audio_rx) => outcome,
                            _ = alive_rx.recv() => Err(anyhow!("cancelled")),
                        }
                    })
                });
            let _ = result_tx.send(outcome);
        });
        let tap: AudioTap = Box::new(move |samples| {
            let _ = audio_tx.try_send(samples.to_vec());
        });
        (
            Self {
                label,
                result,
                _alive: alive_tx,
            },
            tap,
        )
    }

    /// Waits for the transcript. The tap must be dropped first, or the
    /// session keeps waiting for audio until it times out.
    pub fn finish(self) -> Result<String> {
        match self.result.recv_timeout(FINISH_TIMEOUT) {
            Ok(outcome) => outcome,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!("{} did not finish the transcript in time", self.label)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("the {} stream stopped unexpectedly", self.label)
            }
        }
    }
}

async fn stream(
    settings: &AppSettings,
    audio: async_channel::Receiver<Vec<f32>>,
) -> Result<String> {
    let Endpoint {
        label,
        base_url,
        api_key,
        model,
    } = Endpoint::of(settings)?;
    let protocol = Protocol::of(settings.cloud_provider, &model)
        .with_context(|| format!("{model} is not a streaming model"))?;
    let mut request = protocol.url(&base_url, &model)?.into_client_request()?;
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}")).context("the API key is malformed")?,
    );
    let (socket, _) =
        tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(request))
            .await
            .map_err(|_| anyhow!("{label} did not answer"))?
            .map_err(|error| match error {
                // Keeps the provider's reason, which usually says what is wrong.
                // The body arrives with any chunked framing left in, so only its
                // JSON is kept.
                tungstenite::Error::Http(response) => {
                    let body = response
                        .body()
                        .as_deref()
                        .map(String::from_utf8_lossy)
                        .unwrap_or_default();
                    let detail = body
                        .find('{')
                        .zip(body.rfind('}'))
                        .and_then(|(start, end)| body.get(start..=end))
                        .unwrap_or_default()
                        .chars()
                        .take(300)
                        .collect::<String>();
                    anyhow!(
                        "{label} rejected the stream ({}): {detail}",
                        response.status()
                    )
                }
                error => anyhow::Error::new(error).context(format!("could not reach {label}")),
            })?;
    let (mut outgoing, mut incoming) = socket.split();
    let mut send = async |message: Value| {
        outgoing
            .send(Message::text(message.to_string()))
            .await
            .with_context(|| format!("lost the connection to {label}"))
    };
    let rejected = || format!("{label} stopped the stream");

    // Settings sent before the server is ready may be ignored.
    loop {
        let Some(event) = event(incoming.next().await, label)? else {
            continue;
        };
        protocol.read(&event).with_context(rejected)?;
        if event["type"] == "session.created" {
            break;
        }
    }
    send(protocol.setup(&model, &language_hints(settings), &settings.custom_words)).await?;

    let mut resampler = Resampler::new(INPUT_RATE, protocol.sample_rate());
    let mut received = Vec::new();
    let mut converted = Vec::new();
    let mut input_open = true;
    loop {
        tokio::select! {
            samples = audio.recv(), if input_open => {
                let finished = samples.is_err();
                let samples = samples.unwrap_or_default();
                let chunk = match &mut resampler {
                    Some(resampler) => {
                        received.extend_from_slice(&samples);
                        let start = converted.len();
                        resampler.advance(&received, &mut converted, finished);
                        &converted[start..]
                    }
                    None => &samples[..],
                };
                if !chunk.is_empty() {
                    send(protocol.append(chunk)).await?;
                }
                if finished {
                    send(protocol.end()).await?;
                    input_open = false;
                }
            }
            message = incoming.next() => {
                if let Some(event) = event(message, label)?
                    && let Some(text) = protocol.read(&event).with_context(rejected)?
                {
                    return Ok(text);
                }
            }
        }
    }
}

/// Parses a text frame, skipping control frames. A closed stream is an
/// error, since every caller is still waiting on something.
fn event(
    message: Option<Result<Message, tungstenite::Error>>,
    label: &str,
) -> Result<Option<Value>> {
    let message = message
        .with_context(|| format!("{label} closed the stream"))?
        .with_context(|| format!("lost the connection to {label}"))?;
    match message {
        Message::Text(text) => serde_json::from_str(&text)
            .map(Some)
            .with_context(|| format!("{label} sent an unexpected message")),
        Message::Close(frame) => {
            let reason = frame
                .map(|frame| format!(": {}", frame.reason))
                .unwrap_or_default();
            bail!("{label} closed the stream{reason}")
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn only_realtime_models_stream() {
        assert_eq!(
            Protocol::of(CloudProvider::OpenAi, "gpt-live-transcribe"),
            Some(Protocol::OpenAi)
        );
        assert_eq!(Protocol::of(CloudProvider::OpenAi, "gpt-transcribe"), None);
        assert_eq!(
            Protocol::of(CloudProvider::Mistral, "voxtral-mini-transcribe-realtime-3"),
            Some(Protocol::Mistral)
        );
        assert_eq!(
            Protocol::of(CloudProvider::Mistral, "voxtral-mini-transcribe-3"),
            None
        );
        assert_eq!(
            Protocol::of(CloudProvider::Custom, "gpt-live-transcribe"),
            None
        );
    }

    #[test]
    fn openai_live_session_gets_every_language_and_safe_keywords() {
        let setup = Protocol::OpenAi.setup(
            "gpt-live-transcribe",
            &strings(&["ru", "en"]),
            &strings(&["Bisquit", "bad<term"]),
        );
        let input = &setup["session"]["audio"]["input"];
        assert_eq!(input["format"]["rate"], 24_000);
        assert!(input["turn_detection"].is_null());
        assert_eq!(input["transcription"]["languages"], json!(["ru", "en"]));
        assert_eq!(input["transcription"]["keywords"], json!(["Bisquit"]));
    }

    #[test]
    fn mistral_gets_a_language_only_when_it_is_certain() {
        let words = strings(&["GigaAM"]);
        let two = Protocol::Mistral.setup("m", &strings(&["ru", "en"]), &words);
        assert!(two["session"]["language"].is_null());
        assert_eq!(two["session"]["context_bias"], json!(["GigaAM"]));
        let one = Protocol::Mistral.setup("m", &strings(&["ru"]), &words);
        assert_eq!(one["session"]["language"], "ru");
    }

    #[test]
    fn final_events_carry_the_transcript_and_errors_fail() {
        let done = json!({ "type": "transcription.done", "text": " Привет " });
        assert_eq!(
            Protocol::Mistral.read(&done).unwrap().as_deref(),
            Some("Привет")
        );
        let delta = json!({ "type": "transcription.text.delta", "text": "При" });
        assert_eq!(Protocol::Mistral.read(&delta).unwrap(), None);
        let completed = json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello",
        });
        assert_eq!(
            Protocol::OpenAi.read(&completed).unwrap().as_deref(),
            Some("Hello")
        );
        let error = json!({ "type": "error", "error": { "message": "bad key" } });
        assert!(
            Protocol::OpenAi
                .read(&error)
                .unwrap_err()
                .to_string()
                .contains("bad key")
        );
    }
}
