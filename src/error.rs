//! Error type (design §7). Every variant says what went wrong and what to do next (R8).
//!
//! Caller mistakes (`is_caller_error`) come back as `isError: true` tool results with `{error: {code, message, fix}}`,
//! so the model can correct itself; everything else is a JSON-RPC `internal_error` with `data.code`
//! (`SfxError::into_tool_result` does the split in one call).

use std::fmt::Display;
use std::path::{Path, PathBuf};

use rmcp::ErrorData as McpError;
use rmcp::model::CallToolResult;
use serde_json::{Value, json};

/// Kinds a backend may report (`BackendFailure::kind`) that are stable codes of their own (design §7).
const BACKEND_CODES: [&str; 15] = [
    "worker_crashed",
    "out_of_memory",
    "elevenlabs_auth",
    "elevenlabs_permission",
    "elevenlabs_quota",
    "elevenlabs_busy",
    "elevenlabs_validation",
    "network",
    "cancelled",
    // The remaining cloud outcomes keep their own codes too (design §7 lists the common ones).
    "elevenlabs_concurrency",
    "elevenlabs_server",
    "elevenlabs_http",
    "elevenlabs_bad_audio",
    "elevenlabs_bad_response",
    "timeout",
];

/// Details of a backend failure (R8): which stage, what kind, what to try, and the last worker log lines.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("failed during {stage} ({kind}): {message}{}{}", if hint.is_empty() { String::new() } else { format!(". Try: {hint}") }, if log_tail.is_empty() { String::new() } else { format!("\n--- worker log tail ---\n{log_tail}") })]
pub struct BackendFailure {
    /// `loading`, `generating`, `condition`, `sample`, `decode`, `request`, …
    pub stage: String,
    /// `missing_weights`, `nan_output`, `out_of_memory`, `worker_crashed`, `elevenlabs_auth`, `network`, …
    pub kind: String,
    pub message: String,
    pub hint: String,
    pub log_tail: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SfxError {
    #[error("config error in {}: {message}", path.display())]
    Config { path: PathBuf, message: String },

    #[error("invalid `{field}`: {message}")]
    InvalidParams { field: String, message: String },

    #[error("duration_s {duration_s} is outside the range of `{model}` ({min} to {max} s)")]
    DurationOutOfRange {
        model: String,
        duration_s: f64,
        min: f64,
        max: f64,
    },

    /// Two options that can't be combined, or one that doesn't apply to the chosen model (R2).
    #[error("{message}")]
    OptionConflict { message: String, fix: String },

    #[error("unknown model `{id}`. Available: {}", available.join(", "))]
    UnknownModel { id: String, available: Vec<String> },

    #[error("path {} is outside the allowed roots ({})", path.display(), roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "))]
    PathNotAllowed { path: PathBuf, roots: Vec<PathBuf> },

    #[error("{} already exists; existing files are never overwritten", path.display())]
    OutputExists { path: PathBuf },

    /// The model exists but can't run yet; `detail` comes from the registry status (R3). `alternatives` are the other
    /// ready models (R2: never switch to a paid backend on our own, but say what could run).
    #[error("model `{model}` is not ready ({status}): {detail}{}", if alternatives.is_empty() { String::new() } else { format!(" Ready models: {}.", alternatives.join(", ")) })]
    NotReady {
        model: String,
        status: String,
        detail: String,
        alternatives: Vec<String>,
    },

    #[error("model `{model}` needs its licence accepted: {detail}")]
    LicenceNotAccepted { model: String, detail: String },

    #[error("model `{model}` has no credentials: {detail}")]
    CredentialsMissing { model: String, detail: String },

    /// A call that would pass the per-call or daily credit limit, refused before anything is spent (R10).
    #[error("{message}")]
    CreditLimit { message: String, fix: String },

    #[error(
        "not enough free memory to load `{model}`: needs ~{need_mb} MB (model plus headroom), {available_mb} MB available. {advice}"
    )]
    InsufficientMemory {
        model: String,
        need_mb: u64,
        available_mb: u64,
        /// Which heavy processes to close, from the census.
        advice: String,
    },

    #[error("backend `{backend}` is unavailable: {reason}")]
    BackendUnavailable { backend: String, reason: String },

    /// A failure inside the backend, already explained (R8): which stage, what kind, what to try.
    #[error("{backend} {failure}")]
    Backend {
        backend: String,
        failure: Box<BackendFailure>,
    },

    #[error(
        "sound-effect queue is full ({max} waiting). Retry later or cancel a job with cancel_job"
    )]
    QueueFull { max: usize },

    #[error("job was cancelled")]
    Cancelled,

    #[error(
        "unknown job `{0}` (finished jobs expire after job_ttl_s, and jobs don't survive a server restart)"
    )]
    UnknownJob(String),

    #[error("I/O error on {}: {message}", path.display())]
    Io { path: PathBuf, message: String },

    /// The audio pipeline failed on what the backend delivered (a clip too short to loop, an unreadable result, ...).
    #[error("audio processing failed: {message}")]
    Audio { message: String },
}

impl SfxError {
    pub fn config(path: &Path, message: impl Display) -> Self {
        Self::Config {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    pub fn invalid(field: &str, message: impl Display) -> Self {
        Self::InvalidParams {
            field: field.to_string(),
            message: message.to_string(),
        }
    }

    pub fn conflict(message: impl Display, fix: impl Display) -> Self {
        Self::OptionConflict {
            message: message.to_string(),
            fix: fix.to_string(),
        }
    }

    pub fn io(path: &Path, err: impl Display) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        }
    }

    /// A per-call credit limit refusal (R10): `estimate` credits against `limit`.
    pub fn per_call_credit_limit(estimate: u32, limit: u32) -> Self {
        Self::CreditLimit {
            message: format!(
                "this call would cost about {estimate} credits, over the per-call limit of {limit}"
            ),
            fix: "ask for fewer or shorter variations, or raise `per_call_credit_limit` in the [elevenlabs] table of config.toml"
                .into(),
        }
    }

    /// A daily credit limit refusal (R10): `used_today` already spent plus `estimate` against `limit`.
    pub fn daily_credit_limit(estimate: u32, used_today: u32, limit: u32) -> Self {
        Self::CreditLimit {
            message: format!(
                "this call would cost about {estimate} credits and {used_today} were already spent today, over the daily limit of {limit}"
            ),
            fix: "wait until tomorrow (local date), or raise `daily_credit_limit` in the [elevenlabs] table of config.toml"
                .into(),
        }
    }

    /// Stable machine-readable code for structured results.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config_error",
            Self::InvalidParams { .. } => "invalid_params",
            Self::DurationOutOfRange { .. } => "duration_out_of_range",
            Self::OptionConflict { .. } => "option_conflict",
            Self::UnknownModel { .. } => "unknown_model",
            Self::PathNotAllowed { .. } => "path_outside_roots",
            Self::OutputExists { .. } => "output_exists",
            Self::NotReady { .. } => "model_not_ready",
            Self::LicenceNotAccepted { .. } => "licence_not_accepted",
            Self::CredentialsMissing { .. } => "credentials_missing",
            Self::CreditLimit { .. } => "credit_limit",
            Self::InsufficientMemory { .. } => "out_of_memory",
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::Backend { failure, .. } => BACKEND_CODES
                .iter()
                .find(|c| **c == failure.kind)
                .copied()
                .unwrap_or("backend_failed"),
            Self::QueueFull { .. } => "queue_full",
            Self::Cancelled => "cancelled",
            Self::UnknownJob(_) => "unknown_job",
            Self::Io { .. } => "io_error",
            Self::Audio { .. } => "audio_error",
        }
    }

    /// Caller mistakes the model can correct: returned as `isError` tool results, not protocol errors (R6, §7).
    pub fn is_caller_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidParams { .. }
                | Self::DurationOutOfRange { .. }
                | Self::OptionConflict { .. }
                | Self::UnknownModel { .. }
                | Self::PathNotAllowed { .. }
                | Self::OutputExists { .. }
                | Self::NotReady { .. }
                | Self::LicenceNotAccepted { .. }
                | Self::CredentialsMissing { .. }
                | Self::CreditLimit { .. }
                | Self::UnknownJob(_)
        )
    }

    /// The next step, in one sentence (the `fix` of an `isError` result).
    pub fn fix(&self) -> String {
        match self {
            Self::InvalidParams { field, .. } => {
                format!("correct `{field}` and call again")
            }
            Self::DurationOutOfRange { min, max, .. } => {
                format!("pass duration_s between {min} and {max}")
            }
            Self::OptionConflict { fix, .. } | Self::CreditLimit { fix, .. } => fix.clone(),
            Self::UnknownModel { available, .. } => {
                format!("pick one of: {}", available.join(", "))
            }
            Self::PathNotAllowed { roots, .. } => format!(
                "use a path inside one of the allowed roots: {}",
                roots
                    .iter()
                    .map(|r| r.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::OutputExists { .. } => {
                "pick another `name` or `output_dir`, or omit `output_path` so a free name is chosen".into()
            }
            Self::NotReady {
                detail,
                alternatives,
                ..
            } => {
                if alternatives.is_empty() {
                    detail.clone()
                } else {
                    format!("{detail} Or use a ready model: {}.", alternatives.join(", "))
                }
            }
            Self::LicenceNotAccepted { detail, .. } | Self::CredentialsMissing { detail, .. } => {
                detail.clone()
            }
            Self::UnknownJob(_) => "use the job_id returned by generate_sfx with wait=false (finished jobs expire after job_ttl_s)".into(),
            Self::Config { .. } => "fix the file named in the message and restart the server".into(),
            Self::InsufficientMemory { advice, .. } => advice.clone(),
            Self::BackendUnavailable { reason, .. } => reason.clone(),
            Self::Backend { failure, .. } => failure.hint.clone(),
            Self::QueueFull { .. } => "retry later or cancel a job with cancel_job".into(),
            Self::Cancelled => "none: the job was cancelled".into(),
            Self::Io { .. } => "check the path and its permissions".into(),
            Self::Audio { .. } => "try again with another seed or length; if it repeats, report it with the prompt and model".into(),
        }
    }

    /// The message with anything that looks like an API key masked (defence in depth: errors can carry response text
    /// and worker logs, and they end up in tool results, task statuses and logs).
    pub fn scrubbed_message(&self) -> String {
        crate::secrets::scrub(&self.to_string())
    }

    /// `{"error": {"code", "message", "fix"}}`, the body of an `isError` result.
    pub fn error_body(&self) -> Value {
        json!({ "error": { "code": self.code(), "message": self.scrubbed_message(), "fix": crate::secrets::scrub(&self.fix()) } })
    }

    /// The `isError: true` result for a caller error.
    pub fn caller_result(&self) -> CallToolResult {
        CallToolResult::structured_error(self.error_body())
    }

    /// Caller mistakes become `Ok(isError result)`, everything else a JSON-RPC error: the one place tools decide.
    pub fn into_tool_result(self) -> Result<CallToolResult, McpError> {
        if self.is_caller_error() {
            Ok(self.caller_result())
        } else {
            Err(self.into())
        }
    }
}

impl From<SfxError> for McpError {
    fn from(err: SfxError) -> Self {
        let mut data = json!({ "code": err.code() });
        if let SfxError::Backend { backend, failure } = &err {
            data["backend"] = backend.clone().into();
            data["stage"] = failure.stage.clone().into();
            data["kind"] = failure.kind.clone().into();
        }
        let message = err.scrubbed_message();
        if err.is_caller_error() {
            data["fix"] = crate::secrets::scrub(&err.fix()).into();
            McpError::invalid_params(message, Some(data))
        } else {
            McpError::internal_error(message, Some(data))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend(kind: &str, hint: &str, log_tail: &str) -> SfxError {
        SfxError::Backend {
            backend: "sa3_mlx".into(),
            failure: Box::new(BackendFailure {
                stage: "sample".into(),
                kind: kind.into(),
                message: "the worker produced NaN samples".into(),
                hint: hint.into(),
                log_tail: log_tail.into(),
            }),
        }
    }

    #[test]
    fn backend_error_names_stage_kind_and_hint() {
        let msg = backend("nan_output", "try another seed", "").to_string();
        assert_eq!(
            msg,
            "sa3_mlx failed during sample (nan_output): the worker produced NaN samples. Try: try another seed"
        );
        assert!(!msg.contains("log tail"));
        let with_log = backend("nan_output", "", "Traceback ...").to_string();
        assert!(!with_log.contains("Try:"), "{with_log}");
        assert!(with_log.ends_with("--- worker log tail ---\nTraceback ..."));
    }

    #[test]
    fn backend_codes_follow_the_kind() {
        assert_eq!(backend("nan_output", "", "").code(), "backend_failed");
        for kind in BACKEND_CODES {
            assert_eq!(backend(kind, "", "").code(), kind);
        }
        let mcp: McpError = backend("elevenlabs_quota", "", "").into();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INTERNAL_ERROR);
        let data = mcp.data.unwrap();
        assert_eq!(data["code"], "elevenlabs_quota");
        assert_eq!(data["stage"], "sample");
        assert_eq!(data["kind"], "elevenlabs_quota");
        assert_eq!(data["backend"], "sa3_mlx");
    }

    #[test]
    fn caller_errors_are_is_error_results_with_code_message_and_fix() {
        let err = SfxError::DurationOutOfRange {
            model: "sa3-small-sfx".into(),
            duration_s: 45.0,
            min: 0.5,
            max: 30.0,
        };
        assert!(err.is_caller_error());
        let result = err.into_tool_result().expect("a caller error is a result");
        assert_eq!(result.is_error, Some(true));
        let body = result.structured_content.expect("structured body");
        assert_eq!(body["error"]["code"], "duration_out_of_range");
        assert!(
            body["error"]["message"].as_str().unwrap().contains("45"),
            "{body}"
        );
        assert_eq!(body["error"]["fix"], "pass duration_s between 0.5 and 30");
    }

    #[test]
    fn everything_else_is_a_protocol_error() {
        let err = SfxError::InsufficientMemory {
            model: "sa3-small-sfx".into(),
            need_mb: 3700,
            available_mb: 900,
            advice: "Close UnrealEditor (6100 MB).".into(),
        };
        assert!(!err.is_caller_error());
        let msg = err.to_string();
        assert!(msg.contains("3700 MB") && msg.contains("900 MB available"));
        assert!(msg.ends_with("Close UnrealEditor (6100 MB)."), "{msg}");
        let mcp = err.into_tool_result().unwrap_err();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INTERNAL_ERROR);
        assert_eq!(mcp.data.unwrap()["code"], "out_of_memory");
    }

    #[test]
    fn every_design_caller_code_is_a_caller_error() {
        let errors = [
            (SfxError::invalid("prompt", "is required"), "invalid_params"),
            (
                SfxError::DurationOutOfRange {
                    model: "m".into(),
                    duration_s: 99.0,
                    min: 0.5,
                    max: 30.0,
                },
                "duration_out_of_range",
            ),
            (
                SfxError::conflict("loop with trim_silence", "drop one"),
                "option_conflict",
            ),
            (
                SfxError::PathNotAllowed {
                    path: "/x".into(),
                    roots: vec!["/w".into()],
                },
                "path_outside_roots",
            ),
            (
                SfxError::OutputExists {
                    path: "/w/a.wav".into(),
                },
                "output_exists",
            ),
            (
                SfxError::NotReady {
                    model: "m".into(),
                    status: "missing_files".into(),
                    detail: "run: hf download o/r f".into(),
                    alternatives: vec!["other".into()],
                },
                "model_not_ready",
            ),
            (
                SfxError::LicenceNotAccepted {
                    model: "m".into(),
                    detail: "set licence_accepted".into(),
                },
                "licence_not_accepted",
            ),
            (
                SfxError::CredentialsMissing {
                    model: "m".into(),
                    detail: "set ELEVENLABS_API_KEY".into(),
                },
                "credentials_missing",
            ),
            (SfxError::per_call_credit_limit(400, 300), "credit_limit"),
            (
                SfxError::daily_credit_limit(100, 1950, 2000),
                "credit_limit",
            ),
        ];
        for (err, code) in errors {
            assert_eq!(err.code(), code);
            assert!(err.is_caller_error(), "{code}");
            let body = err.error_body();
            assert_eq!(body["error"]["code"], code);
            assert!(!body["error"]["fix"].as_str().unwrap().is_empty(), "{code}");
            let mcp: McpError = err.into();
            assert_eq!(mcp.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        }
    }

    #[test]
    fn not_ready_names_the_ready_alternatives() {
        let err = SfxError::NotReady {
            model: "sa3-small-sfx".into(),
            status: "backend_unavailable".into(),
            detail: "MLX needs Apple Silicon".into(),
            alternatives: vec!["elevenlabs-sfx-v2".into()],
        };
        assert!(
            err.to_string()
                .ends_with("Ready models: elevenlabs-sfx-v2."),
            "{err}"
        );
        assert!(err.fix().contains("elevenlabs-sfx-v2"));
    }

    #[test]
    fn credit_limits_say_how_to_raise_them() {
        let per_call = SfxError::per_call_credit_limit(400, 300).to_string();
        assert!(
            per_call.contains("400") && per_call.contains("300"),
            "{per_call}"
        );
        assert!(
            SfxError::per_call_credit_limit(400, 300)
                .fix()
                .contains("per_call_credit_limit")
        );
        let daily = SfxError::daily_credit_limit(100, 1950, 2000);
        assert!(daily.to_string().contains("1950"), "{daily}");
        assert!(daily.fix().contains("daily_credit_limit"));
    }

    #[test]
    fn codes_are_stable() {
        assert_eq!(SfxError::Cancelled.code(), "cancelled");
        assert_eq!(SfxError::QueueFull { max: 4 }.code(), "queue_full");
        assert_eq!(SfxError::UnknownJob("j".into()).code(), "unknown_job");
        assert!(SfxError::UnknownJob("j".into()).is_caller_error());
        assert!(!SfxError::Cancelled.is_caller_error());
    }
}
