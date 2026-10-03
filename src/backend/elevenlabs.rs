//! ElevenLabs Sound Effects over HTTP (design §4.3, requirements R8 to R10).
//!
//! `POST /v1/sound-generation` returns headerless 16-bit little-endian stereo PCM, with the cost in the
//! `character-cost` header. `GET /v1/user/subscription` shows the plan (a key without `user_read` gets a
//! [`ElevenLabsError::Permission`] there, which is a normal outcome).
//!
//! Every error message passes through [`crate::secrets::scrub_with_key`], and the key is only ever placed in a
//! header marked sensitive.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use tokio::sync::Semaphore;

use super::{
    AccountError, AccountSnapshot, Availability, BackendError, BackendRequest, ProgressSink,
    RawClip, Runs, SfxBackend, Stage,
};
use crate::audio::Clip;
use crate::config::ElevenLabsConfig;
use crate::ledger::{Entry as LedgerEntry, Ledger};
use crate::registry::{BackendOptions, ModelSpec};
use crate::secrets::{ApiKey, scrub_with_key};

pub const DEFAULT_BASE_URL: &str = "https://api.elevenlabs.io";
pub const DEFAULT_MODEL_ID: &str = "eleven_text_to_sound_v2";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Total attempts per request (the first try plus two retries).
pub const MAX_ATTEMPTS: u32 = 3;
const DEFAULT_RETRY_DELAYS: [Duration; 2] = [Duration::from_secs(1), Duration::from_secs(3)];
/// A `Retry-After` longer than this is not waited for; the error is returned instead.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);
const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Raw PCM formats that work on every plan (measured 2026-10-03). Both are s16le stereo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PcmFormat {
    #[default]
    Pcm44100,
    Pcm48000,
}

impl PcmFormat {
    /// The `output_format` query value.
    pub fn query_value(self) -> &'static str {
        match self {
            PcmFormat::Pcm44100 => "pcm_44100",
            PcmFormat::Pcm48000 => "pcm_48000",
        }
    }

    pub fn sample_rate(self) -> u32 {
        match self {
            PcmFormat::Pcm44100 => 44_100,
            PcmFormat::Pcm48000 => 48_000,
        }
    }
}

/// One sound-generation request.
#[derive(Debug, Clone, PartialEq)]
pub struct SfxRequest {
    pub text: String,
    /// `eleven_text_to_sound_v2` (the only model that supports `loop_`).
    pub model_id: String,
    /// 0.5 to 30 seconds; `None` lets the model pick (a flat 50 credits).
    pub duration_seconds: Option<f64>,
    /// 0 to 1 (API default 0.3).
    pub prompt_influence: f64,
    /// Seamless loop (v2 only).
    pub loop_: bool,
    pub output_format: PcmFormat,
}

impl SfxRequest {
    pub fn new(text: impl Into<String>) -> Self {
        SfxRequest {
            text: text.into(),
            model_id: DEFAULT_MODEL_ID.to_string(),
            duration_seconds: None,
            prompt_influence: 0.3,
            loop_: false,
            output_format: PcmFormat::default(),
        }
    }

    /// Checks the ranges the API enforces, so a bad request costs no round trip. Returns
    /// [`ElevenLabsError::Validation`].
    pub fn validate(&self) -> Result<(), ElevenLabsError> {
        let bad = |field: &str, message: &str| {
            Err(ElevenLabsError::Validation {
                field: field.to_string(),
                message: message.to_string(),
            })
        };
        if self.text.trim().is_empty() {
            return bad("text", "the prompt is empty");
        }
        if let Some(d) = self.duration_seconds
            && !(d.is_finite() && (0.5..=30.0).contains(&d))
        {
            return bad("duration_seconds", "must be between 0.5 and 30 seconds");
        }
        if !(self.prompt_influence.is_finite() && (0.0..=1.0).contains(&self.prompt_influence)) {
            return bad("prompt_influence", "must be between 0 and 1");
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct SoundGenerationBody<'a> {
    text: &'a str,
    model_id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration_seconds: Option<f64>,
    prompt_influence: f64,
    #[serde(rename = "loop")]
    loop_: bool,
}

/// The audio from one successful call.
#[derive(Clone, PartialEq)]
pub struct SfxAudio {
    /// Interleaved 16-bit little-endian samples (length is a multiple of 4).
    pub pcm_s16le: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u16,
    /// From the `character-cost` header of the *successful* response only. Failed attempts carry no cost.
    pub credits: Option<u32>,
    /// The `request-id` header, if any.
    pub request_id: Option<String>,
    /// How many HTTP attempts it took (1 means no retry).
    pub attempts: u32,
}

impl std::fmt::Debug for SfxAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SfxAudio")
            .field("pcm_bytes", &self.pcm_s16le.len())
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("credits", &self.credits)
            .field("request_id", &self.request_id)
            .field("attempts", &self.attempts)
            .finish()
    }
}

/// `GET /v1/user/subscription`, the parts we use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountInfo {
    pub tier: String,
    pub status: String,
    pub character_count: u64,
    pub character_limit: u64,
    pub next_reset_unix: Option<i64>,
}

/// Why a call failed. Messages are scrubbed of keys. `code()` is the tool-result / JSON-RPC `data.code`; `fix()` says
/// what to do next (R8).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ElevenLabsError {
    /// 401 (invalid key) or an otherwise unclassified 403.
    #[error("ElevenLabs rejected the API key: {message}")]
    Auth { message: String },
    /// 401 `missing_permissions`: the key is scoped and lacks `permission` (e.g. `user_read`).
    #[error("the API key is missing the `{permission}` permission: {message}")]
    Permission { permission: String, message: String },
    /// 402, or `quota_exceeded`: out of credits or at a plan limit.
    #[error("ElevenLabs credits or plan limit reached: {message}")]
    Quota { message: String },
    /// 429 `system_busy`, still failing after the retries.
    #[error("ElevenLabs is busy: {message}")]
    Busy { message: String },
    /// 429 `too_many_concurrent_requests`. Never retried here: the caller's semaphore handles it.
    #[error("too many concurrent ElevenLabs requests for this plan: {message}")]
    Concurrency { message: String },
    /// 422, or caught before sending. `field` is the offending parameter (from `loc`).
    #[error("ElevenLabs rejected the `{field}` parameter: {message}")]
    Validation { field: String, message: String },
    /// 5xx after the retries.
    #[error("ElevenLabs server error (HTTP {status}): {message}")]
    Server { status: u16, message: String },
    /// Any other 4xx.
    #[error("ElevenLabs returned HTTP {status}: {message}")]
    Http { status: u16, message: String },
    /// Could not reach ElevenLabs, or the connection broke.
    #[error("network error talking to ElevenLabs: {0}")]
    Network(String),
    #[error("the ElevenLabs request timed out")]
    Timeout,
    /// A 200 that isn't usable audio.
    #[error("ElevenLabs returned unusable audio: {0}")]
    BadAudio(String),
    /// A success response that isn't the JSON we expected.
    #[error("unexpected ElevenLabs response: {0}")]
    BadResponse(String),
    #[error("the ElevenLabs request was cancelled")]
    Cancelled,
}

impl ElevenLabsError {
    /// Stable machine-readable code.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Auth { .. } => "elevenlabs_auth",
            Self::Permission { .. } => "elevenlabs_permission",
            Self::Quota { .. } => "elevenlabs_quota",
            Self::Busy { .. } => "elevenlabs_busy",
            Self::Concurrency { .. } => "elevenlabs_concurrency",
            Self::Validation { .. } => "elevenlabs_validation",
            Self::Server { .. } => "elevenlabs_server",
            Self::Http { .. } => "elevenlabs_http",
            Self::Network(_) => "network",
            Self::Timeout => "timeout",
            Self::BadAudio(_) => "elevenlabs_bad_audio",
            Self::BadResponse(_) => "elevenlabs_bad_response",
            Self::Cancelled => "cancelled",
        }
    }

    /// The next step for the user or agent.
    pub fn fix(&self) -> String {
        match self {
            Self::Auth { .. } => "Check the ElevenLabs API key: it may be wrong or revoked. Set it in the environment \
                variable or Keychain item named in the [elevenlabs] config table, or create a new key in the \
                ElevenLabs dashboard."
                .into(),
            Self::Permission { permission, .. } => format!(
                "Give the API key the `{permission}` permission in the ElevenLabs dashboard (API keys, edit), or \
                 use a key that has it."
            ),
            Self::Quota { .. } => "The account is out of credits or at its plan limit. Wait for the monthly reset, \
                upgrade the plan, or use the local model instead. Check `account_status`."
                .into(),
            Self::Busy { .. } => "ElevenLabs is overloaded. It was retried 3 times; try again in a minute.".into(),
            Self::Concurrency { .. } => "The plan allows only a few simultaneous requests (Free 2, Starter 3, \
                Creator 5, Pro 10). Wait for running jobs, or lower `concurrency` in the [elevenlabs] config table."
                .into(),
            Self::Validation { field, .. } => {
                format!("Change the `{field}` argument and call again. No credits were spent.")
            }
            Self::Server { .. } => "ElevenLabs had a server error on all 3 attempts. Try again later \
                (status.elevenlabs.io)."
                .into(),
            Self::Http { .. } => "ElevenLabs refused the request. Check the message; no credits were spent.".into(),
            Self::Network(_) => "Check the internet connection, proxy and firewall, then try again.".into(),
            Self::Timeout => "ElevenLabs did not answer in time. Try again, or raise `timeout_s` in the \
                [elevenlabs] config table. A timed-out request may still have used credits: see `account_status` \
                or the credit ledger."
                .into(),
            Self::BadAudio(_) => "ElevenLabs returned something that isn't audio. Try again; if it repeats, report it."
                .into(),
            Self::BadResponse(_) => "ElevenLabs answered in an unexpected format. Try again; if it repeats, report it."
                .into(),
            Self::Cancelled => "The request was cancelled and nothing was written. If ElevenLabs had already \
                finished, the credits were still spent."
                .into(),
        }
    }
}

/// What to do after a failed attempt.
enum Retry {
    No,
    Backoff,
    After(Duration),
}

struct Failure {
    error: ElevenLabsError,
    retry: Retry,
}

impl Failure {
    fn no_retry(error: ElevenLabsError) -> Self {
        Failure {
            error,
            retry: Retry::No,
        }
    }
}

/// Client for the ElevenLabs HTTP API. Cheap to clone.
#[derive(Debug, Clone)]
pub struct ElevenLabsClient {
    base_url: String,
    http: reqwest::Client,
    retry_delays: [Duration; 2],
}

impl ElevenLabsClient {
    /// `base_url` is normally [`DEFAULT_BASE_URL`]; `timeout` covers one whole request including the body
    /// (default [`DEFAULT_TIMEOUT`]). Connecting gets its own 10 s limit.
    pub fn new(base_url: impl Into<String>, timeout: Duration) -> Result<Self, ElevenLabsError> {
        let http = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(CONNECT_TIMEOUT)
            .user_agent(concat!("mcp-sfxgen/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| ElevenLabsError::Network(scrub_with_key(&error_chain(&e), None)))?;
        Ok(ElevenLabsClient {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            http,
            retry_delays: DEFAULT_RETRY_DELAYS,
        })
    }

    /// Replaces the delays before the 2nd and 3rd attempts (default 1 s, 3 s; ±25 % jitter is still applied).
    /// Meant for tests.
    pub fn with_retry_delays(mut self, delays: [Duration; 2]) -> Self {
        self.retry_delays = delays;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Generates one sound. Retries `system_busy`, 5xx, connect errors and timeouts (see [`MAX_ATTEMPTS`]); never
    /// 401/402/403/422 or other 4xx. Cancelling drops the in-flight request.
    pub async fn generate(
        &self,
        key: &ApiKey,
        req: &SfxRequest,
        cancel: &CancellationToken,
    ) -> Result<SfxAudio, ElevenLabsError> {
        req.validate()?;
        let url = format!("{}/v1/sound-generation", self.base_url);
        let body = SoundGenerationBody {
            text: &req.text,
            model_id: &req.model_id,
            duration_seconds: req.duration_seconds,
            prompt_influence: req.prompt_influence,
            loop_: req.loop_,
        };
        let (response, attempts) = self
            .call(key, cancel, || {
                self.http
                    .post(&url)
                    .query(&[("output_format", req.output_format.query_value())])
                    .json(&body)
            })
            .await?;

        let headers = response.headers().clone();
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();
        // `audio/*` is what we expect. Reject the shapes a failed call really has (JSON, text, HTML) rather than
        // insisting on a content type we haven't seen for raw PCM, because the credits are spent by this point.
        if content_type.starts_with("text/")
            || content_type.contains("json")
            || content_type.contains("html")
        {
            return Err(ElevenLabsError::BadAudio(scrub_with_key(
                &format!("expected audio but the content type was `{content_type}`"),
                Some(key),
            )));
        }
        if response.status() != StatusCode::OK {
            return Err(ElevenLabsError::BadAudio(format!(
                "expected HTTP 200 with audio, got {}",
                response.status().as_u16()
            )));
        }

        let credits =
            header_str(&headers, "character-cost").and_then(|v| v.trim().parse::<u32>().ok());
        let request_id = header_str(&headers, "request-id")
            .or_else(|| header_str(&headers, "x-request-id"))
            .map(|v| scrub_with_key(v, Some(key)));

        let pcm = self
            .read_body(response, key, cancel, MAX_AUDIO_BYTES)
            .await
            .map_err(|f| f.error)?;
        if pcm.is_empty() {
            return Err(ElevenLabsError::BadAudio("the audio was empty".into()));
        }
        if pcm.len() % 4 != 0 {
            return Err(ElevenLabsError::BadAudio(format!(
                "{} bytes is not a whole number of 16-bit stereo frames",
                pcm.len()
            )));
        }
        Ok(SfxAudio {
            pcm_s16le: pcm,
            sample_rate: req.output_format.sample_rate(),
            channels: 2,
            credits,
            request_id,
            attempts,
        })
    }

    /// Reads the plan and credits. A key without `user_read` gives [`ElevenLabsError::Permission`], which callers
    /// turn into "not readable".
    pub async fn account(&self, key: &ApiKey) -> Result<AccountInfo, ElevenLabsError> {
        self.account_cancellable(key, &CancellationToken::new())
            .await
    }

    /// [`ElevenLabsClient::account`] with a cancellation token.
    pub async fn account_cancellable(
        &self,
        key: &ApiKey,
        cancel: &CancellationToken,
    ) -> Result<AccountInfo, ElevenLabsError> {
        let url = format!("{}/v1/user/subscription", self.base_url);
        let (response, _) = self.call(key, cancel, || self.http.get(&url)).await?;
        let bytes = self
            .read_body(response, key, cancel, MAX_ERROR_BODY_BYTES)
            .await
            .map_err(|f| f.error)?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|_| {
            ElevenLabsError::BadResponse("the subscription reply was not JSON".into())
        })?;
        let text = |name: &str| {
            value
                .get(name)
                .and_then(Value::as_str)
                .map(|s| scrub_with_key(s, Some(key)))
                .unwrap_or_default()
        };
        let number = |name: &str| value.get(name).and_then(Value::as_u64).unwrap_or(0);
        Ok(AccountInfo {
            tier: text("tier"),
            status: text("status"),
            character_count: number("character_count"),
            character_limit: number("character_limit"),
            next_reset_unix: value
                .get("next_character_count_reset_unix")
                .and_then(Value::as_i64),
        })
    }

    /// Sends the request built by `build`, with retries. Returns the first 2xx response and the attempt count.
    async fn call(
        &self,
        key: &ApiKey,
        cancel: &CancellationToken,
        build: impl Fn() -> RequestBuilder,
    ) -> Result<(Response, u32), ElevenLabsError> {
        let mut key_header = HeaderValue::from_str(key.expose()).map_err(|_| ElevenLabsError::Auth {
            message: "the API key contains characters that can't be sent in a header (stray newline?)".into(),
        })?;
        key_header.set_sensitive(true);

        let mut attempt: u32 = 0;
        loop {
            if cancel.is_cancelled() {
                return Err(ElevenLabsError::Cancelled);
            }
            attempt += 1;
            let request = build().header("xi-api-key", key_header.clone());
            let outcome = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(ElevenLabsError::Cancelled),
                r = self.attempt_once(request, key) => r,
            };
            let failure = match outcome {
                Ok(response) => return Ok((response, attempt)),
                Err(failure) => failure,
            };

            let delay = if attempt >= MAX_ATTEMPTS {
                None
            } else {
                match failure.retry {
                    Retry::No => None,
                    Retry::Backoff => Some(jitter(self.retry_delays[(attempt - 1) as usize])),
                    Retry::After(d) if d <= MAX_RETRY_AFTER => Some(d),
                    Retry::After(_) => None,
                }
            };
            tracing::debug!(
                code = failure.error.code(),
                attempt,
                retry_in_ms = delay.map(|d| d.as_millis() as u64),
                "ElevenLabs attempt failed"
            );
            let Some(delay) = delay else {
                return Err(failure.error);
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(ElevenLabsError::Cancelled),
                _ = tokio::time::sleep(delay) => {}
            }
        }
    }

    /// One HTTP exchange: a 2xx response comes back as `Ok`, everything else is classified.
    async fn attempt_once(
        &self,
        request: RequestBuilder,
        key: &ApiKey,
    ) -> Result<Response, Failure> {
        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => return Err(transport_failure(&e, key)),
        };
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = parse_retry_after(response.headers());
        let body = self
            .read_body(
                response,
                key,
                &CancellationToken::new(),
                MAX_ERROR_BODY_BYTES,
            )
            .await
            .unwrap_or_default();
        Err(classify(status, retry_after, &body, key))
    }

    /// Reads a body up to `limit` bytes, abortable by `cancel`.
    async fn read_body(
        &self,
        mut response: Response,
        key: &ApiKey,
        cancel: &CancellationToken,
        limit: usize,
    ) -> Result<Vec<u8>, Failure> {
        let mut body = Vec::new();
        loop {
            let chunk = tokio::select! {
                biased;
                _ = cancel.cancelled() => return Err(Failure::no_retry(ElevenLabsError::Cancelled)),
                c = response.chunk() => c,
            };
            match chunk {
                Ok(Some(bytes)) => {
                    if body.len() + bytes.len() > limit {
                        return Err(Failure::no_retry(ElevenLabsError::BadAudio(format!(
                            "the reply is larger than {} MiB",
                            limit / (1024 * 1024)
                        ))));
                    }
                    body.extend_from_slice(&bytes);
                }
                Ok(None) => return Ok(body),
                // A broken body follows a 200 that the server may already have charged for, so it is not retried.
                Err(e) => return Err(Failure::no_retry(transport_error(&e, key))),
            }
        }
    }
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let seconds: f64 = header_str(headers, "retry-after")?.trim().parse().ok()?;
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds.min(86_400.0)))
}

/// ±25 % random jitter.
fn jitter(base: Duration) -> Duration {
    use rand::Rng;
    base.mul_f64(rand::rng().random_range(0.75..=1.25))
}

fn error_chain(e: &dyn std::error::Error) -> String {
    let mut text = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        text.push_str(": ");
        text.push_str(&s.to_string());
        source = s.source();
    }
    text
}

fn transport_error(e: &reqwest::Error, key: &ApiKey) -> ElevenLabsError {
    if e.is_timeout() {
        ElevenLabsError::Timeout
    } else {
        // The URL has no secrets, but the chain can carry arbitrary text, so scrub it.
        ElevenLabsError::Network(scrub_with_key(&error_chain(e), Some(key)))
    }
}

fn transport_failure(e: &reqwest::Error, key: &ApiKey) -> Failure {
    // No response arrived, so no audio was produced: timeouts and connection errors are retried.
    Failure {
        error: transport_error(e, key),
        retry: Retry::Backoff,
    }
}

/// What `detail` said, in either of the two shapes ElevenLabs uses.
#[derive(Debug, Default, PartialEq)]
struct Detail {
    /// `status` (or `code`) of the object shape, lower-cased.
    kind: String,
    message: String,
    /// For validation errors: the parameter name.
    field: Option<String>,
}

fn parse_detail(body: &[u8]) -> Detail {
    let value: Option<Value> = serde_json::from_slice(body).ok();
    let mut detail = Detail::default();
    let root_message = value
        .as_ref()
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str);
    match value.as_ref().and_then(|v| v.get("detail")) {
        Some(Value::Object(o)) => {
            let text = |name: &str| o.get(name).and_then(Value::as_str);
            detail.kind = text("status")
                .or_else(|| text("code"))
                .unwrap_or_default()
                .to_ascii_lowercase();
            detail.message = text("message").unwrap_or_default().to_string();
        }
        Some(Value::Array(items)) => {
            let messages: Vec<&str> = items
                .iter()
                .filter_map(|i| i.get("msg").and_then(Value::as_str))
                .collect();
            detail.message = messages.join("; ");
            detail.field = items.first().and_then(|i| {
                i.get("loc")
                    .and_then(Value::as_array)?
                    .iter()
                    .rev()
                    .find_map(|part| part.as_str().filter(|s| *s != "body" && *s != "query"))
                    .map(str::to_string)
            });
        }
        Some(Value::String(s)) => detail.message = s.clone(),
        _ => {
            detail.message = root_message.map(str::to_string).unwrap_or_else(|| {
                String::from_utf8_lossy(&body[..body.len().min(200)])
                    .trim()
                    .to_string()
            });
        }
    }
    detail
}

/// `"... is missing the permission user_read to ..."` gives `user_read`.
fn parse_permission(message: &str) -> Option<String> {
    let lower = message.to_ascii_lowercase();
    let at = lower.find("permission")? + "permission".len();
    let rest = &message[at..];
    let rest = rest.strip_prefix(['s', 'S']).unwrap_or(rest);
    let rest = rest.trim_start_matches(|c: char| !c.is_ascii_alphanumeric());
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

fn classify(
    status: StatusCode,
    retry_after: Option<Duration>,
    body: &[u8],
    key: &ApiKey,
) -> Failure {
    let detail = parse_detail(body);
    let code = status.as_u16();
    let message = {
        let m = scrub_with_key(detail.message.trim(), Some(key));
        if m.is_empty() {
            "(no detail)".to_string()
        } else {
            m
        }
    };
    let kind = detail.kind.as_str();
    let backoff = || retry_after.map_or(Retry::Backoff, Retry::After);

    if kind == "missing_permissions" || (code == 401 && message.contains("missing the permission"))
    {
        let permission = parse_permission(&message).unwrap_or_else(|| "unknown".to_string());
        return Failure::no_retry(ElevenLabsError::Permission {
            permission,
            message,
        });
    }
    if code == 402 || (code < 500 && kind.contains("quota_exceeded")) {
        return Failure::no_retry(ElevenLabsError::Quota { message });
    }
    match code {
        401 | 403 => Failure::no_retry(ElevenLabsError::Auth { message }),
        422 => Failure::no_retry(ElevenLabsError::Validation {
            field: scrub_with_key(detail.field.as_deref().unwrap_or("request"), Some(key)),
            message,
        }),
        429 if kind == "system_busy" => Failure {
            error: ElevenLabsError::Busy { message },
            retry: backoff(),
        },
        429 if kind == "too_many_concurrent_requests" => {
            Failure::no_retry(ElevenLabsError::Concurrency { message })
        }
        // A rate limit we don't recognise: report it as busy, but don't spend retries on it.
        429 => Failure::no_retry(ElevenLabsError::Busy { message }),
        500..=599 => Failure {
            error: ElevenLabsError::Server {
                status: code,
                message,
            },
            retry: backoff(),
        },
        _ => Failure::no_retry(ElevenLabsError::Http {
            status: code,
            message,
        }),
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The `SfxBackend` adapter: variations as concurrent requests, key lookup, error mapping.

/// Looks up the API key (default: environment variable, then the OS credential store).
type KeyProvider =
    Arc<dyn Fn() -> Result<Option<ApiKey>, crate::secrets::SecretError> + Send + Sync>;

/// How long a looked-up key is reused (the lookup may run the `security` CLI).
const KEY_CACHE: Duration = Duration::from_secs(60);
/// Times a `too_many_concurrent_requests` answer is waited out before giving up.
const CONCURRENCY_RETRIES: u32 = 5;
const DEFAULT_CONCURRENCY_WAIT: Duration = Duration::from_millis(750);
/// The account answer is reused this long (the call is free, but there's no reason to repeat it).
const ACCOUNT_CACHE: Duration = Duration::from_secs(60);

/// ElevenLabs as an [`SfxBackend`]: cloud lane, `provider` "elevenlabs".
pub struct ElevenLabsBackend {
    config: ElevenLabsConfig,
    client: ElevenLabsClient,
    semaphore: Arc<Semaphore>,
    key_provider: KeyProvider,
    key_cache: Mutex<Option<(Instant, ApiKey)>>,
    account_cache: Mutex<Option<(Instant, Result<AccountSnapshot, String>)>>,
    ledger: Option<Ledger>,
    concurrency_wait: Duration,
}

impl ElevenLabsBackend {
    pub fn new(config: ElevenLabsConfig) -> Result<Self, ElevenLabsError> {
        Self::with_base_url(config, DEFAULT_BASE_URL)
    }

    /// For tests and proxies: a different API origin.
    pub fn with_base_url(
        config: ElevenLabsConfig,
        base_url: impl Into<String>,
    ) -> Result<Self, ElevenLabsError> {
        let client = ElevenLabsClient::new(base_url, Duration::from_secs(config.timeout_s))?;
        let (env, service) = (config.api_key_env.clone(), config.keychain_service.clone());
        Ok(Self {
            semaphore: Arc::new(Semaphore::new(config.concurrency.max(1))),
            config,
            client,
            key_provider: Arc::new(move || crate::secrets::resolve(&env, &service)),
            key_cache: Mutex::new(None),
            account_cache: Mutex::new(None),
            ledger: None,
            concurrency_wait: DEFAULT_CONCURRENCY_WAIT,
        })
    }

    /// Replace the key lookup (tests).
    pub fn with_key_provider(
        mut self,
        provider: impl Fn() -> Result<Option<ApiKey>, crate::secrets::SecretError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.key_provider = Arc::new(provider);
        self
    }

    /// Shorter retry delays inside the client (tests).
    pub fn with_retry_delays(mut self, delays: [Duration; 2]) -> Self {
        self.client = self.client.with_retry_delays(delays);
        self
    }

    /// How long a `too_many_concurrent_requests` answer waits before the request goes back through the semaphore.
    pub fn with_concurrency_wait(mut self, wait: Duration) -> Self {
        self.concurrency_wait = wait;
        self
    }

    /// Record credits spent on clips that had to be thrown away (a sibling variation failed, or the job was cancelled).
    /// Clips that make it to the job finisher are recorded there, with their file names.
    pub fn with_ledger(mut self, ledger: Ledger) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// The key, looked up at call time and cached for 60 s. `Ok(None)` when there is none.
    async fn key(&self) -> Result<Option<ApiKey>, BackendError> {
        if let Some((at, key)) = self.key_cache.lock().unwrap().as_ref()
            && at.elapsed() < KEY_CACHE
        {
            return Ok(Some(key.clone()));
        }
        let provider = self.key_provider.clone();
        let found = tokio::task::spawn_blocking(move || provider())
            .await
            .map_err(|e| {
                BackendError::failed("request", "credential_lookup", e.to_string(), "try again")
            })?
            .map_err(|e| {
                BackendError::failed(
                    "request",
                    "credential_lookup",
                    scrub_with_key(&e.to_string(), None),
                    "check that the Keychain is unlocked, or set the key in the environment variable named in the [elevenlabs] config table",
                )
            })?;
        if let Some(key) = &found {
            *self.key_cache.lock().unwrap() = Some((Instant::now(), key.clone()));
        }
        Ok(found)
    }

    async fn key_or_unavailable(&self) -> Result<ApiKey, BackendError> {
        self.key().await?.ok_or_else(|| BackendError::Unavailable {
            reason: self.config.credentials_hint(),
        })
    }

    /// The scrubbed, explained failure for an API error.
    fn failure(&self, error: ElevenLabsError, key: &ApiKey, stage: &str) -> BackendError {
        if matches!(error, ElevenLabsError::Cancelled) {
            return BackendError::Cancelled;
        }
        BackendError::failed(
            stage,
            error.code(),
            scrub_with_key(&error.to_string(), Some(key)),
            scrub_with_key(&error.fix(), Some(key)),
        )
    }

    fn record_discarded(
        &self,
        model: &str,
        credits: &[(Option<u32>, Option<f64>, Option<String>)],
    ) {
        let Some(ledger) = &self.ledger else { return };
        for (credits, duration_s, request_id) in credits {
            let Some(credits) = credits else { continue };
            let mut entry = LedgerEntry::now(model, *credits);
            entry.duration_s = *duration_s;
            entry.request_id = request_id.clone();
            if let Err(e) = ledger.append(&entry) {
                tracing::warn!(error = %scrub_with_key(&e.to_string(), None), "couldn't record discarded credits");
            }
        }
    }
}

/// One variation: wait for a semaphore slot, call the API, and put the request back in the queue when the plan's
/// concurrency limit answers anyway (another process may share the key).
async fn one_variation(
    client: ElevenLabsClient,
    key: ApiKey,
    req: SfxRequest,
    semaphore: Arc<Semaphore>,
    wait: Duration,
    cancel: CancellationToken,
) -> Result<SfxAudio, ElevenLabsError> {
    let mut retries = 0;
    loop {
        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Err(ElevenLabsError::Cancelled),
            p = semaphore.acquire() => p.map_err(|_| ElevenLabsError::Cancelled)?,
        };
        let outcome = client.generate(&key, &req, &cancel).await;
        drop(permit);
        match outcome {
            Err(ElevenLabsError::Concurrency { .. }) if retries < CONCURRENCY_RETRIES => {
                retries += 1;
                tokio::select! {
                    biased;
                    _ = cancel.cancelled() => return Err(ElevenLabsError::Cancelled),
                    _ = tokio::time::sleep(wait * retries) => {}
                }
            }
            other => return other,
        }
    }
}

#[async_trait::async_trait]
impl SfxBackend for ElevenLabsBackend {
    fn id(&self) -> &'static str {
        "elevenlabs"
    }

    fn runs(&self) -> Runs {
        Runs::Cloud
    }

    fn provider(&self) -> Option<&'static str> {
        Some("elevenlabs")
    }

    /// Always ready when compiled in: a missing key is the registry's `needs_credentials` status, not unavailability.
    async fn availability(&self, _model: &ModelSpec) -> Availability {
        Availability::Ready
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        request: &BackendRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError> {
        let BackendOptions::Elevenlabs {
            model_id,
            prompt_influence,
        } = &model.options
        else {
            return Err(BackendError::Unavailable {
                reason: format!("model `{}` has no elevenlabs options", model.id),
            });
        };
        let key = self.key_or_unavailable().await?;
        let format = if request.sample_rate == 48_000 {
            PcmFormat::Pcm48000
        } else {
            PcmFormat::Pcm44100
        };
        let mut api = SfxRequest::new(request.prompt.clone());
        api.model_id = model_id.clone();
        api.duration_seconds = request.duration_s;
        api.prompt_influence = request.prompt_influence.unwrap_or(*prompt_influence);
        api.loop_ = request.loop_;
        api.output_format = format;
        // Fail on the caller's mistakes before any task starts (no credits are spent).
        api.validate()
            .map_err(|e| self.failure(e, &key, "request"))?;

        let total = request.variations.max(1);
        progress.report(Stage::Generating { n: 0, of: total });
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..total {
            let task = one_variation(
                self.client.clone(),
                key.clone(),
                api.clone(),
                self.semaphore.clone(),
                self.concurrency_wait,
                cancel.clone(),
            );
            tasks.spawn(async move { (index, task.await) });
        }

        let mut done: Vec<Option<SfxAudio>> = (0..total).map(|_| None).collect();
        let mut first_error: Option<ElevenLabsError> = None;
        let mut finished = 0u32;
        while let Some(joined) = tasks.join_next().await {
            let (index, outcome) = match joined {
                Ok(pair) => pair,
                Err(e) => {
                    first_error.get_or_insert(ElevenLabsError::Network(format!(
                        "a request task failed: {e}"
                    )));
                    cancel_rest(&mut tasks);
                    continue;
                }
            };
            match outcome {
                Ok(audio) => {
                    finished += 1;
                    progress.report(Stage::Generating {
                        n: finished,
                        of: total,
                    });
                    done[index as usize] = Some(audio);
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                    cancel_rest(&mut tasks);
                }
            }
        }

        if let Some(error) = first_error {
            // What the finished siblings cost is spent either way: keep it in the ledger.
            let spent: Vec<_> = done
                .iter()
                .flatten()
                .map(|a| (a.credits, request.duration_s, a.request_id.clone()))
                .collect();
            self.record_discarded(&model.id, &spent);
            return Err(if cancel.is_cancelled() {
                BackendError::Cancelled
            } else {
                self.failure(error, &key, "request")
            });
        }

        let mut clips = Vec::with_capacity(done.len());
        for (index, audio) in done.into_iter().enumerate() {
            let audio = audio.expect("every variation finished");
            let clip = Clip::from_s16le_bytes(&audio.pcm_s16le, audio.sample_rate, audio.channels)
                .map_err(|e| {
                    self.failure(ElevenLabsError::BadAudio(e.to_string()), &key, "request")
                })?;
            clips.push(RawClip {
                samples: clip.samples,
                sample_rate: audio.sample_rate,
                channels: audio.channels,
                seed: None,
                source_format: format.query_value().to_string(),
                credits: audio.credits,
                details: serde_json::json!({
                    "request_id": audio.request_id,
                    "attempts": audio.attempts,
                    "batch": { "size": total, "index": index },
                }),
            });
        }
        Ok(clips)
    }

    /// Nothing is resident.
    async fn unload(&self) {}

    fn loaded_model(&self) -> Option<String> {
        None
    }

    /// The plan and credits, when the key can read them. Cached for 60 s; free.
    async fn account(&self) -> Option<Result<AccountSnapshot, AccountError>> {
        let cached = self.account_cache.lock().unwrap().clone();
        if let Some((at, result)) = cached
            && at.elapsed() < ACCOUNT_CACHE
        {
            return Some(result.map_err(|reason| AccountError::NotReadable { reason }));
        }
        let key = match self.key().await {
            Ok(Some(key)) => key,
            Ok(None) => {
                return Some(Err(AccountError::NotReadable {
                    reason: self.config.credentials_hint(),
                }));
            }
            Err(e) => return Some(Err(AccountError::Failed(e))),
        };
        match self.client.account(&key).await {
            Ok(info) => {
                let snapshot = AccountSnapshot::from(info);
                *self.account_cache.lock().unwrap() = Some((Instant::now(), Ok(snapshot.clone())));
                Some(Ok(snapshot))
            }
            Err(ElevenLabsError::Permission { permission, .. }) => {
                let reason = format!(
                    "the API key can't read the account (it lacks the `{permission}` permission); the plan comes from config and spend from the local ledger"
                );
                *self.account_cache.lock().unwrap() = Some((Instant::now(), Err(reason.clone())));
                Some(Err(AccountError::NotReadable { reason }))
            }
            Err(e) => Some(Err(AccountError::Failed(self.failure(e, &key, "account")))),
        }
    }
}

fn cancel_rest(tasks: &mut tokio::task::JoinSet<(u32, Result<SfxAudio, ElevenLabsError>)>) {
    // Requests still in flight are dropped (their credits, if any, are unknown); queued ones never start.
    tasks.abort_all();
}

impl From<AccountInfo> for AccountSnapshot {
    fn from(info: AccountInfo) -> Self {
        AccountSnapshot {
            plan: info.tier,
            status: info.status,
            credits_used: info.character_count,
            credits_limit: info.character_limit,
            credits_remaining: info.character_limit.saturating_sub(info.character_count),
            resets_at: info
                .next_reset_unix
                .and_then(|t| chrono::DateTime::from_timestamp(t, 0))
                .map(|t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "sk_0123456789abcdef0123456789abcdef";

    fn key() -> ApiKey {
        ApiKey::new(KEY)
    }

    fn classify_body(status: u16, body: &str) -> Failure {
        classify(
            StatusCode::from_u16(status).unwrap(),
            None,
            body.as_bytes(),
            &key(),
        )
    }

    #[test]
    fn detail_object_shape() {
        let d = parse_detail(
            br#"{"detail":{"type":"authentication_error","code":"missing_permissions","message":"The API key you used is missing the permission user_read to execute this operation.","status":"missing_permissions","request_id":"abc"}}"#,
        );
        assert_eq!(d.kind, "missing_permissions");
        assert!(d.message.contains("user_read"));
    }

    #[test]
    fn detail_array_shape_gives_the_field() {
        let d = parse_detail(
            br#"{"detail":[{"type":"less_than_equal","loc":["body","duration_seconds"],"msg":"Input should be less than or equal to 30","input":99}]}"#,
        );
        assert_eq!(d.field.as_deref(), Some("duration_seconds"));
        assert_eq!(d.message, "Input should be less than or equal to 30");
        // Integer path parts are skipped; only the last name counts.
        let d = parse_detail(br#"{"detail":[{"loc":["body","items",3,"text"],"msg":"bad"}]}"#);
        assert_eq!(d.field.as_deref(), Some("text"));
    }

    #[test]
    fn detail_other_shapes() {
        assert_eq!(
            parse_detail(br#"{"detail":"plain words"}"#).message,
            "plain words"
        );
        assert_eq!(
            parse_detail(br#"{"message":"top level"}"#).message,
            "top level"
        );
        assert_eq!(parse_detail(b"Bad Gateway").message, "Bad Gateway");
        assert_eq!(parse_detail(b"").message, "");
        assert!(!parse_detail(b"\xff\xfe").message.is_empty());
    }

    #[test]
    fn permission_names() {
        assert_eq!(
            parse_permission("The API key you used is missing the permission user_read to execute this operation.").as_deref(),
            Some("user_read")
        );
        assert_eq!(
            parse_permission("missing permissions: sound_generation, user_read").as_deref(),
            Some("sound_generation")
        );
        assert_eq!(parse_permission("nothing relevant"), None);
        assert_eq!(parse_permission("permission"), None);
    }

    #[test]
    fn classification_table() {
        let f = classify_body(
            401,
            r#"{"detail":{"status":"missing_permissions","message":"missing the permission user_read to execute"}}"#,
        );
        assert!(
            matches!(&f.error, ElevenLabsError::Permission { permission, .. } if permission == "user_read")
        );
        assert!(matches!(f.retry, Retry::No));

        // Permission recognised from the message alone as well.
        let f = classify_body(
            401,
            r#"{"detail":{"status":"x","message":"The API key is missing the permission user_read."}}"#,
        );
        assert!(
            matches!(&f.error, ElevenLabsError::Permission { permission, .. } if permission == "user_read")
        );

        let f = classify_body(
            401,
            r#"{"detail":{"status":"invalid_api_key","message":"Invalid API key"}}"#,
        );
        assert!(matches!(f.error, ElevenLabsError::Auth { .. }));
        assert!(matches!(f.retry, Retry::No));

        let f = classify_body(402, "{}");
        assert!(matches!(f.error, ElevenLabsError::Quota { .. }));
        let f = classify_body(
            401,
            r#"{"detail":{"status":"quota_exceeded","message":"You have 5 credits remaining, while 50 credits are required."}}"#,
        );
        assert!(
            matches!(&f.error, ElevenLabsError::Quota { message } if message.contains("5 credits"))
        );
        let f = classify_body(
            403,
            r#"{"detail":{"status":"quota_exceeded","message":"x"}}"#,
        );
        assert!(matches!(f.error, ElevenLabsError::Quota { .. }));
        let f = classify_body(403, r#"{"detail":{"status":"forbidden","message":"x"}}"#);
        assert!(matches!(f.error, ElevenLabsError::Auth { .. }));

        let f = classify_body(
            422,
            r#"{"detail":[{"loc":["body","prompt_influence"],"msg":"too big"}]}"#,
        );
        assert!(
            matches!(&f.error, ElevenLabsError::Validation { field, message } if field == "prompt_influence" && message == "too big")
        );
        let f = classify_body(422, r#"{"detail":{"message":"nope"}}"#);
        assert!(
            matches!(&f.error, ElevenLabsError::Validation { field, .. } if field == "request")
        );

        let f = classify_body(
            429,
            r#"{"detail":{"status":"system_busy","message":"busy"}}"#,
        );
        assert!(matches!(f.error, ElevenLabsError::Busy { .. }));
        assert!(matches!(f.retry, Retry::Backoff));
        let f = classify(
            StatusCode::TOO_MANY_REQUESTS,
            Some(Duration::from_secs(2)),
            br#"{"detail":{"status":"system_busy","message":"busy"}}"#,
            &key(),
        );
        assert!(matches!(f.retry, Retry::After(d) if d == Duration::from_secs(2)));
        let f = classify_body(
            429,
            r#"{"detail":{"status":"too_many_concurrent_requests","message":"slow down"}}"#,
        );
        assert!(matches!(f.error, ElevenLabsError::Concurrency { .. }));
        assert!(matches!(f.retry, Retry::No));
        let f = classify_body(429, r#"{"detail":{"status":"rate_limit","message":"x"}}"#);
        assert!(matches!(f.retry, Retry::No));

        for s in [500, 502, 503] {
            let f = classify_body(s, "oops");
            assert!(matches!(f.error, ElevenLabsError::Server { status, .. } if status == s));
            assert!(matches!(f.retry, Retry::Backoff));
        }
        for s in [400, 404, 409] {
            let f = classify_body(s, "{}");
            assert!(matches!(f.error, ElevenLabsError::Http { .. }), "{s}");
            assert!(matches!(f.retry, Retry::No));
        }
    }

    #[test]
    fn classified_messages_never_hold_the_key() {
        let body = format!(
            r#"{{"detail":{{"status":"invalid_api_key","message":"Invalid API key {KEY} (and {KEY})"}}}}"#
        );
        let f = classify_body(401, &body);
        let shown = format!("{} | {:?} | {}", f.error, f.error, f.error.fix());
        assert!(!shown.contains("0123456789abcdef"), "{shown}");
        assert!(shown.contains("sk_<redacted>"));

        // Also a key that doesn't look like `sk_...`.
        let odd = ApiKey::new("hunter2-hunter2-hunter2");
        let f = classify(
            StatusCode::INTERNAL_SERVER_ERROR,
            None,
            b"echo hunter2-hunter2-hunter2",
            &odd,
        );
        assert!(!format!("{:?}", f.error).contains("hunter2"));
    }

    #[test]
    fn every_error_has_a_code_and_a_fix() {
        let all = [
            ElevenLabsError::Auth {
                message: "m".into(),
            },
            ElevenLabsError::Permission {
                permission: "user_read".into(),
                message: "m".into(),
            },
            ElevenLabsError::Quota {
                message: "m".into(),
            },
            ElevenLabsError::Busy {
                message: "m".into(),
            },
            ElevenLabsError::Concurrency {
                message: "m".into(),
            },
            ElevenLabsError::Validation {
                field: "f".into(),
                message: "m".into(),
            },
            ElevenLabsError::Server {
                status: 500,
                message: "m".into(),
            },
            ElevenLabsError::Http {
                status: 404,
                message: "m".into(),
            },
            ElevenLabsError::Network("n".into()),
            ElevenLabsError::Timeout,
            ElevenLabsError::BadAudio("b".into()),
            ElevenLabsError::BadResponse("b".into()),
            ElevenLabsError::Cancelled,
        ];
        let mut codes = std::collections::HashSet::new();
        for e in &all {
            assert!(!e.fix().is_empty() && !e.to_string().is_empty());
            assert!(codes.insert(e.code()), "duplicate code {}", e.code());
        }
        assert_eq!(all[0].code(), "elevenlabs_auth");
        assert_eq!(all[1].code(), "elevenlabs_permission");
        assert_eq!(all[2].code(), "elevenlabs_quota");
        assert_eq!(all[3].code(), "elevenlabs_busy");
        assert_eq!(all[5].code(), "elevenlabs_validation");
        assert_eq!(all[8].code(), "network");
        assert!(all[1].fix().contains("user_read"));
    }

    #[test]
    fn request_validation() {
        let ok = SfxRequest::new("a horn");
        assert!(ok.validate().is_ok());
        let field = |r: SfxRequest| match r.validate() {
            Err(ElevenLabsError::Validation { field, .. }) => field,
            other => panic!("expected validation error, got {other:?}"),
        };
        assert_eq!(field(SfxRequest::new("  ")), "text");
        for d in [0.4, 30.1, f64::NAN, -1.0] {
            let mut r = SfxRequest::new("x");
            r.duration_seconds = Some(d);
            assert_eq!(field(r), "duration_seconds", "{d}");
        }
        for d in [0.5, 30.0] {
            let mut r = SfxRequest::new("x");
            r.duration_seconds = Some(d);
            assert!(r.validate().is_ok());
        }
        let mut r = SfxRequest::new("x");
        r.prompt_influence = 1.01;
        assert_eq!(field(r), "prompt_influence");
    }

    #[test]
    fn jitter_stays_within_25_percent() {
        let base = Duration::from_millis(1000);
        for _ in 0..500 {
            let d = jitter(base);
            assert!(
                d >= Duration::from_millis(750) && d <= Duration::from_millis(1250),
                "{d:?}"
            );
        }
    }

    #[test]
    fn retry_after_parsing() {
        let mut h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
        h.insert("retry-after", HeaderValue::from_static("3"));
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(3)));
        h.insert(
            "retry-after",
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&h), None);
        h.insert("retry-after", HeaderValue::from_static("-4"));
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn formats_and_debug() {
        assert_eq!(PcmFormat::Pcm44100.query_value(), "pcm_44100");
        assert_eq!(PcmFormat::Pcm48000.query_value(), "pcm_48000");
        assert_eq!(PcmFormat::Pcm48000.sample_rate(), 48_000);
        let audio = SfxAudio {
            pcm_s16le: vec![0; 8],
            sample_rate: 44_100,
            channels: 2,
            credits: Some(10),
            request_id: None,
            attempts: 1,
        };
        assert!(format!("{audio:?}").contains("pcm_bytes: 8"));
    }
}
