//! Sound-effect backends behind one trait (design §4.1; R2, R4).
//!
//! A backend turns a prompt into raw clips in memory (interleaved f32). It never writes the files the caller sees:
//! post-processing and writing are the job finisher's (`jobs::JobFinisher`), so every backend gets the same
//! limiter, naming, sidecar and provenance. Backends report progress through a `ProgressSink` and honour the
//! cancellation token; scratch files go under `BackendRequest::out_dir`, which the job manager deletes afterwards.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

pub use crate::error::BackendFailure;
use crate::error::SfxError;
use crate::registry::ModelSpec;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
pub mod process;
#[cfg(feature = "sa3_mlx")]
pub mod sa3_mlx;

/// Where a backend does its work. Cloud backends skip the GPU lock and the memory preflight and run in their own lane.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Runs {
    #[default]
    Local,
    Cloud,
}

impl std::fmt::Display for Runs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Local => "local",
            Self::Cloud => "cloud",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability {
    Ready,
    /// Why it can't run here, including how to fix it.
    Unavailable {
        reason: String,
    },
}

impl Availability {
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Ready => None,
            Self::Unavailable { reason } => Some(reason),
        }
    }
}

/// Where a job is (R4): queued, waiting for the GPU lock, loading, generating n of N, post-processing, writing.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Stage {
    Queued {
        position: usize,
    },
    WaitingGpuLock {
        holder: Option<String>,
    },
    Loading,
    /// Clip `n` of `of` (`of` = variations, or a backend's own count such as sampling steps).
    Generating {
        n: u32,
        of: u32,
    },
    PostProcessing,
    Writing,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued { position } => write!(f, "queued (position {position})"),
            Self::WaitingGpuLock { holder } => {
                f.write_str(&crate::gpu_lock::waiting_message_for(holder.as_deref()))
            }
            Self::Loading => f.write_str("loading model"),
            Self::Generating { n, of } => write!(f, "generating {n} of {of}"),
            Self::PostProcessing => f.write_str("post-processing"),
            Self::Writing => f.write_str("writing files"),
        }
    }
}

/// Cheap clonable progress callback. Backends call it; the job manager records it.
#[derive(Clone)]
pub struct ProgressSink(Arc<dyn Fn(Stage) + Send + Sync>);

impl ProgressSink {
    pub fn new(f: impl Fn(Stage) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn noop() -> Self {
        Self::new(|_| {})
    }

    pub fn report(&self, stage: Stage) {
        (self.0)(stage)
    }
}

/// One generation, resolved (defaults filled in, parameters checked against the model).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BackendRequest {
    pub prompt: String,
    /// `None` = the model picks the length (ElevenLabs only).
    pub duration_s: Option<f64>,
    /// Clips to make (1 to 4).
    pub variations: u32,
    /// `None` = random, always reported back in `RawClip::seed`; ignored by backends without seeds.
    pub seed: Option<u64>,
    /// Ask the backend for a seamless loop (native-loop backends only; the local loop is made in post).
    pub loop_: bool,
    /// Local backends: sampler steps (`None` = the model's default).
    pub steps: Option<u32>,
    /// Cloud backends: 0 to 1 (`None` = the model's default).
    pub prompt_influence: Option<f64>,
    /// The sample rate to ask for where the backend can deliver it (44100 or 48000); the clip reports what it got.
    pub sample_rate: u32,
    /// Scratch folder for raw temporary files. Set by the job manager (created before, deleted after the job).
    pub out_dir: PathBuf,
}

/// What a backend hands back for one clip: interleaved f32 samples and where they came from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawClip {
    /// Interleaved, `channels` per frame, nominally within -1..1 (a cloud clip can exceed it before limiting).
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    /// The seed actually used; `None` for backends without seeds (ElevenLabs).
    pub seed: Option<u64>,
    /// What the backend delivered, e.g. `pcm_44100`, `f32` (recorded in the sidecar).
    pub source_format: String,
    /// Credits the provider charged for this clip (from its response), if any.
    pub credits: Option<u32>,
    /// Backend-specific provenance for the sidecar (batch size and index, timings, versions, …).
    pub details: serde_json::Value,
}

impl RawClip {
    pub fn frames(&self) -> usize {
        self.samples.len() / usize::from(self.channels.max(1))
    }

    pub fn duration_s(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames() as f64 / f64::from(self.sample_rate)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    Cancelled,
    Unavailable {
        reason: String,
    },
    /// Explained failure (R8).
    Failed(BackendFailure),
}

impl BackendError {
    pub fn failed(
        stage: &str,
        kind: &str,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::Failed(BackendFailure {
            stage: stage.to_string(),
            kind: kind.to_string(),
            message: message.into(),
            hint: hint.into(),
            log_tail: String::new(),
        })
    }

    /// A failure with the last worker log lines attached.
    pub fn failed_with_log(mut self, log_tail: impl Into<String>) -> Self {
        if let Self::Failed(f) = &mut self {
            f.log_tail = log_tail.into();
        }
        self
    }

    pub fn into_error(self, backend: &str) -> SfxError {
        match self {
            Self::Cancelled => SfxError::Cancelled,
            Self::Unavailable { reason } => SfxError::BackendUnavailable {
                backend: backend.to_string(),
                reason,
            },
            Self::Failed(failure) => SfxError::Backend {
                backend: backend.to_string(),
                failure: Box::new(failure),
            },
        }
    }
}

/// A cloud account's plan and credits, as far as the key can read them (`account_status`).
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
pub struct AccountSnapshot {
    /// The plan's name as the provider reports it (`free`, `starter`, ...).
    pub plan: String,
    pub status: String,
    pub credits_used: u64,
    pub credits_limit: u64,
    pub credits_remaining: u64,
    /// When the credits reset, RFC 3339 UTC, if the provider says.
    pub resets_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AccountError {
    /// A normal outcome: no key, or a key without the permission to read the account.
    NotReadable {
        reason: String,
    },
    Failed(BackendError),
}

#[async_trait::async_trait]
pub trait SfxBackend: Send + Sync {
    fn id(&self) -> &'static str;

    /// Local machine or a cloud service (the job lane).
    fn runs(&self) -> Runs;

    /// File- and platform-level check only; never loads weights and never makes a paid call.
    async fn availability(&self, model: &ModelSpec) -> Availability;

    /// Make `request.variations` clips. Must honour `cancel`, report progress, and keep temporary files under
    /// `request.out_dir`. Cloud backends may run variations concurrently inside this call.
    async fn generate(
        &self,
        model: &ModelSpec,
        request: &BackendRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError>;

    /// Free memory now (stop a kept-alive worker). Idempotent.
    async fn unload(&self);

    /// Id of the model currently resident, if any.
    fn loaded_model(&self) -> Option<String>;

    /// Who receives prompts, for cloud backends (`ElevenLabs`).
    fn provider(&self) -> Option<&'static str> {
        None
    }

    /// Cloud backends: the account's plan and credits. `None` = this backend has no account.
    async fn account(&self) -> Option<Result<AccountSnapshot, AccountError>> {
        None
    }
}

/// Whether this build includes the backend: the Cargo feature of the same name.
pub fn compiled_in(backend: &str) -> bool {
    match backend {
        "sa3_mlx" => cfg!(feature = "sa3_mlx"),
        "elevenlabs" => cfg!(feature = "elevenlabs"),
        b if b.starts_with("mock") => cfg!(any(test, feature = "mock")),
        _ => false,
    }
}

/// Backends keyed by id.
#[derive(Clone, Default)]
pub struct BackendRegistry {
    backends: HashMap<String, Arc<dyn SfxBackend>>,
}

impl BackendRegistry {
    pub fn insert(&mut self, backend: Arc<dyn SfxBackend>) {
        self.backends.insert(backend.id().to_string(), backend);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn SfxBackend>> {
        self.backends.get(id).cloned()
    }

    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn SfxBackend>> {
        self.backends.values()
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.backends.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Availability of a model's backend, with a clear reason when this build doesn't include it
    /// (or includes it but nothing registered it).
    pub async fn availability_of(&self, model: &ModelSpec) -> Availability {
        match self.get(&model.backend) {
            Some(b) => b.availability(model).await,
            None => Availability::Unavailable {
                reason: if compiled_in(&model.backend) {
                    format!(
                        "backend `{}` is compiled in but wasn't registered by the server",
                        model.backend
                    )
                } else {
                    format!(
                        "backend `{}` is not in this build (built with: {}). Rebuild with `--features {}`",
                        model.backend,
                        if self.backends.is_empty() {
                            "none".to_string()
                        } else {
                            self.ids().join(", ")
                        },
                        model.backend
                    )
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;
    use crate::registry::ModelRegistry;

    #[test]
    fn stage_messages() {
        assert_eq!(
            Stage::Generating { n: 2, of: 4 }.to_string(),
            "generating 2 of 4"
        );
        assert_eq!(
            Stage::Queued { position: 2 }.to_string(),
            "queued (position 2)"
        );
        assert_eq!(Stage::Loading.to_string(), "loading model");
        assert_eq!(Stage::PostProcessing.to_string(), "post-processing");
        assert_eq!(Stage::Writing.to_string(), "writing files");
        assert_eq!(
            Stage::WaitingGpuLock {
                holder: Some("mcp-meshgen (pid 7)".into())
            }
            .to_string(),
            "waiting for another local model (mcp-meshgen (pid 7))"
        );
        let json = serde_json::to_value(Stage::Generating { n: 1, of: 4 }).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"stage": "generating", "n": 1, "of": 4})
        );
        let json = serde_json::to_value(Stage::WaitingGpuLock { holder: None }).unwrap();
        assert_eq!(json["stage"], "waiting_gpu_lock");
    }

    #[test]
    fn failures_map_to_explained_errors() {
        let err = BackendError::failed("sample", "nan_output", "NaN samples", "try another seed")
            .failed_with_log("last line")
            .into_error("sa3_mlx");
        assert_eq!(err.code(), "backend_failed");
        let msg = err.to_string();
        assert!(msg.contains("sample (nan_output)"), "{msg}");
        assert!(msg.contains("last line"), "{msg}");
        assert_eq!(BackendError::Cancelled.into_error("x").code(), "cancelled");
        let err = BackendError::Unavailable {
            reason: "no venv".into(),
        }
        .into_error("sa3_mlx");
        assert_eq!(err.code(), "backend_unavailable");
        let err = BackendError::failed("request", "elevenlabs_auth", "401", "check the key")
            .into_error("elevenlabs");
        assert_eq!(err.code(), "elevenlabs_auth");
    }

    #[test]
    fn raw_clip_durations() {
        let clip = RawClip {
            samples: vec![0.0; 88_200],
            sample_rate: 44_100,
            channels: 2,
            ..Default::default()
        };
        assert_eq!(clip.frames(), 44_100);
        assert!((clip.duration_s() - 1.0).abs() < 1e-9);
        assert_eq!(RawClip::default().duration_s(), 0.0);
    }

    #[tokio::test]
    async fn missing_backend_is_unavailable_with_a_reason() {
        let reg = BackendRegistry::default();
        let sa3 = ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("sa3-small-sfx")
            .unwrap()
            .clone();
        let reason = reg
            .availability_of(&sa3)
            .await
            .reason()
            .unwrap()
            .to_string();
        if cfg!(feature = "sa3_mlx") {
            assert!(
                reason.contains("compiled in but wasn't registered"),
                "{reason}"
            );
        } else {
            assert!(
                reason.contains("not in this build") && reason.contains("--features sa3_mlx"),
                "{reason}"
            );
        }
        let mut other = sa3;
        other.backend = "future_backend".into();
        let reason = reg
            .availability_of(&other)
            .await
            .reason()
            .unwrap()
            .to_string();
        assert!(
            reason.contains("not in this build") && reason.contains("--features future_backend"),
            "{reason}"
        );
    }

    #[test]
    fn compiled_in_follows_the_features() {
        assert_eq!(compiled_in("sa3_mlx"), cfg!(feature = "sa3_mlx"));
        assert_eq!(compiled_in("elevenlabs"), cfg!(feature = "elevenlabs"));
        assert!(compiled_in("mock-cloud"));
        assert!(!compiled_in("openai"));
    }

    #[test]
    fn runs_display_and_serde() {
        assert_eq!(Runs::Cloud.to_string(), "cloud");
        assert_eq!(serde_json::to_value(Runs::Local).unwrap(), "local");
        assert_eq!(
            serde_json::from_str::<Runs>("\"cloud\"").unwrap(),
            Runs::Cloud
        );
        assert_eq!(Runs::default(), Runs::Local);
    }
}
