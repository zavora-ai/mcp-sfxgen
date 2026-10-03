//! MCP surface (design §5; R1 to R4, R6, R10): `generate_sfx` (a Task tool), `get_job`, `cancel_job`, `list_models`,
//! `account_status` and `unload_models`.
//!
//! Every tool declares a typed output schema (`result.rs`). Caller mistakes come back as `isError` results with
//! `{error: {code, message, fix}}`; everything else is a JSON-RPC error with `data.code` (`SfxError::into_tool_result`).
//! A cloud call is checked against the credit limits before it is queued, so a refused call sends nothing.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use adk_mcp_sdk::{HealthCheck, HealthStatus};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Resource};
use rmcp::{ErrorData as McpError, tool, tool_router};
use serde::Serialize;
use serde_json::Value;

use crate::backend::{AccountError, Availability, BackendRegistry, Runs};
use crate::config::{Config, default_hf_cache};
use crate::error::SfxError;
use crate::finish::{SfxFinisher, commercial_for};
use crate::gpu_lock::GpuLock;
use crate::heavy::{ProcessLister, SystemProcesses};
use crate::jobs::{CancelOnDrop, JobDeps, JobManager, JobRequest, JobSnapshot, JobStatus};
use crate::ledger::{CostRates, Ledger, check_limits, estimate};
use crate::memory::{MemoryProbe, SystemMemory};
use crate::output::PathPolicy;
use crate::registry::{ModelRegistry, ModelSpec, ModelStatus, StatusContext};
use crate::request::{JobIdInput, ResolvedRequest, SfxRequest};
use crate::result::{
    AccountStatusOutput, ErrorBody, ErrorDetail, FileResult, GenerateOutput, GenerateResult,
    JobOutput, JobStarted, LedgerTotals, ListModelsOutput, ModelEntry, UnloadOutput,
};
use crate::secrets::{self, KeySource};

/// How long a key-presence check is reused (on macOS it runs the `security` CLI).
const PRESENCE_CACHE: Duration = Duration::from_secs(60);
/// MCP task lifetime (one hour, like `job_ttl_s`'s default).
const TASK_TTL_MS: u64 = 3_600_000;

struct State {
    cfg: Config,
    registry: ModelRegistry,
    jobs: JobManager,
    memory: Arc<dyn MemoryProbe>,
    paths: PathPolicy,
    ledger: Ledger,
    presence: Mutex<Option<(Instant, bool)>>,
}

#[derive(Clone)]
pub struct SfxGenServer {
    state: Arc<State>,
}

/// Backends compiled into this build (R2): local Stable Audio 3, ElevenLabs, and the mocks under the `mock` feature.
pub fn default_backends(cfg: &Config) -> BackendRegistry {
    #[allow(unused_mut)]
    let mut backends = BackendRegistry::default();
    #[cfg(feature = "sa3_mlx")]
    backends.insert(Arc::new(crate::backend::sa3_mlx::Sa3MlxBackend::new(
        cfg.sa3_mlx.clone(),
        cfg.hf_cache.clone(),
        cfg.app_cache_dir(),
        cfg.idle_unload_s,
    )));
    #[cfg(feature = "elevenlabs")]
    match crate::backend::elevenlabs::ElevenLabsBackend::new(cfg.elevenlabs.clone()) {
        Ok(backend) => {
            backends.insert(Arc::new(
                backend.with_ledger(Ledger::open(cfg.ledger_path())),
            ));
        }
        Err(e) => {
            tracing::error!(error = %secrets::scrub(&e.to_string()), "ElevenLabs backend not started")
        }
    }
    #[cfg(feature = "mock")]
    {
        let (local, cloud) = crate::backend::mock::env_backends();
        backends.insert(Arc::new(local));
        backends.insert(Arc::new(cloud));
    }
    let _ = cfg;
    backends
}

impl SfxGenServer {
    /// Must be called inside a tokio runtime (spawns the job runners).
    pub fn new(
        cfg: Config,
        registry: ModelRegistry,
        backends: BackendRegistry,
        memory: Arc<dyn MemoryProbe>,
        processes: Arc<dyn ProcessLister>,
    ) -> Self {
        let jobs = JobManager::new(JobDeps {
            backends,
            gpu_lock: GpuLock::new(cfg.gpu_lock_path()),
            memory: memory.clone(),
            processes,
            heavy_processes: cfg.heavy_processes.clone(),
            memory_headroom_mb: cfg.memory_headroom_mb,
            queue_depth: cfg.queue_depth,
            job_ttl: Duration::from_secs(cfg.job_ttl_s),
            idle_unload: Duration::from_secs(cfg.idle_unload_s),
            scratch_dir: cfg.scratch_dir(),
        });
        Self {
            state: Arc::new(State {
                paths: PathPolicy::from_config(&cfg),
                ledger: Ledger::open(cfg.ledger_path()),
                cfg,
                registry,
                jobs,
                memory,
                presence: Mutex::new(None),
            }),
        }
    }

    /// Load config and registry from the default location and build every compiled backend.
    pub fn from_default_config() -> Result<Self, SfxError> {
        let cfg = Config::load_or_init(&Config::default_dir())?;
        let registry = ModelRegistry::load(&cfg.models_path())?;
        let backends = default_backends(&cfg);
        Ok(Self::new(
            cfg,
            registry,
            backends,
            Arc::new(SystemMemory::default()),
            Arc::new(SystemProcesses::default()),
        ))
    }

    pub fn config(&self) -> &Config {
        &self.state.cfg
    }

    /// Stop everything, even mid-job: cancel running jobs (which kills a local worker's process tree) and stop kept
    /// workers. The worker runs in its own process group, so exiting the server alone wouldn't take it down.
    pub async fn shutdown(&self) {
        self.state.jobs.cancel_all();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while self.state.jobs.busy() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        for backend in self.state.jobs.deps().backends.all() {
            backend.unload().await;
        }
    }

    /// Whether an ElevenLabs key exists in the environment or the credential store. Only presence is ever reported;
    /// the lookup may run the `security` CLI, so the answer is reused for a minute.
    async fn credentials_present(&self) -> bool {
        if let Some((at, present)) = *self.state.presence.lock().unwrap()
            && at.elapsed() < PRESENCE_CACHE
        {
            return present;
        }
        let el = &self.state.cfg.elevenlabs;
        let (env, service) = (el.api_key_env.clone(), el.keychain_service.clone());
        let present = tokio::task::spawn_blocking(move || {
            secrets::presence(&env, &service) != KeySource::None
        })
        .await
        .unwrap_or(false);
        *self.state.presence.lock().unwrap() = Some((Instant::now(), present));
        present
    }

    /// The status of one model right now (R3): never loads weights, never makes a paid call.
    async fn status_of(&self, model: &ModelSpec) -> ModelStatus {
        let st = &self.state;
        let availability = st.jobs.deps().backends.availability_of(model).await;
        let default_cache = default_hf_cache();
        let credentials_present =
            model.options.kind() == "elevenlabs" && self.credentials_present().await;
        model.status(&StatusContext {
            hf_cache: &st.cfg.hf_cache,
            default_cache: &default_cache,
            backend_unavailable: availability.reason(),
            licence_accepted: st.cfg.sa3_mlx.licence_is_accepted(),
            credentials_present,
            credentials_hint: &st.cfg.elevenlabs.credentials_hint(),
        })
    }

    /// Ids of the other models that could run now (named in a refusal; one is never picked for the caller).
    async fn ready_models_except(&self, id: &str) -> Vec<String> {
        let mut ready = Vec::new();
        for model in self.state.registry.models() {
            if model.id != id && self.status_of(model).await.is_ready() {
                ready.push(model.id.clone());
            }
        }
        ready
    }

    async fn model_entry(&self, model: &ModelSpec) -> ModelEntry {
        let st = &self.state;
        let status = self.status_of(model).await;
        let loaded = st
            .jobs
            .deps()
            .backends
            .get(&model.backend)
            .is_some_and(|b| b.loaded_model().as_deref() == Some(model.id.as_str()));
        let runs = model.runs();
        // The plan can narrow what the licence allows: ElevenLabs' Free plan is non-commercial (R10).
        let (commercial_use, commercial_note) = commercial_for(model, &st.cfg);
        let mut entry = ModelEntry {
            id: model.id.clone(),
            name: model.name.clone(),
            backend: model.backend.clone(),
            runs,
            status: status.code().to_string(),
            detail: None,
            loaded,
            duration_s: model.duration_s,
            native_loop: model.native_loop,
            typical_s: model.typical_s,
            licences: model.licence.clone(),
            commercial_use,
            commercial_note,
            memory_mb: (runs == Runs::Local).then_some(model.memory_mb),
            needs_mb_free: model.needs_mb_free(st.cfg.memory_headroom_mb),
            privacy: model.privacy.clone(),
            // Presence only: the key itself never leaves `secrets`.
            credentials_present: (model.options.kind() == "elevenlabs")
                .then_some(self.credentials_present().await),
            missing: None,
            download_mb: None,
            accept: None,
            how: None,
            hint: None,
            reason: None,
        };
        if !status.is_ready() {
            entry.detail = Some(status.detail());
        }
        match status {
            ModelStatus::Ready => {}
            ModelStatus::MissingFiles {
                missing,
                download_mb,
            } => {
                entry.missing = Some(missing);
                entry.download_mb = Some(download_mb);
            }
            ModelStatus::NeedsLicenceAcceptance {
                accept,
                how,
                missing,
                download_mb,
            } => {
                entry.accept = Some(accept);
                entry.how = Some(how);
                entry.missing = Some(missing);
                entry.download_mb = Some(download_mb);
            }
            ModelStatus::NeedsCredentials { hint } => entry.hint = Some(hint),
            ModelStatus::BackendUnavailable { reason } => entry.reason = Some(reason),
        }
        entry
    }

    /// Validate (R1, R2, R7, R12), check the model's status (R3) and the credit limits (R10), then queue.
    async fn submit(&self, input: &SfxRequest) -> Result<(JobSnapshot, ResolvedRequest), SfxError> {
        let st = &self.state;
        let resolved = input.resolve(&st.registry, &st.cfg, &st.paths)?;
        let status = self.status_of(&resolved.model).await;
        if !status.is_ready() {
            let alternatives = self.ready_models_except(&resolved.model.id).await;
            return Err(status.not_ready_error(&resolved.model.id, alternatives));
        }
        if !resolved.target.numbered {
            // An exact path that is taken fails now rather than when the job starts.
            let wav = resolved
                .target
                .dir
                .join(format!("{}.wav", resolved.target.stem));
            if wav.exists() {
                return Err(SfxError::OutputExists { path: wav });
            }
        }
        if resolved.model.runs() == Runs::Cloud {
            self.check_credit_limits(&resolved)?;
        }
        let finisher = Arc::new(SfxFinisher::new(&resolved, &st.cfg, st.ledger.clone()));
        let snapshot = st.jobs.submit(JobRequest {
            model: resolved.model.clone(),
            request: resolved.backend.clone(),
            finisher,
        })?;
        Ok((snapshot, resolved))
    }

    /// The cost guard (R10): the estimate against the per-call and daily limits, before anything is sent.
    fn check_credit_limits(&self, resolved: &ResolvedRequest) -> Result<(), SfxError> {
        let el = &self.state.cfg.elevenlabs;
        let rates = CostRates {
            credits_per_second: el.credits_per_second,
            credits_auto_duration: el.credits_auto_duration,
        };
        let cost = estimate(
            resolved.backend.duration_s,
            resolved.backend.variations,
            &rates,
        );
        let today = self
            .state
            .ledger
            .today_total()
            .map_err(|e| SfxError::CreditLimit {
                message: format!("today's spend is unknown: {e}"),
                fix: "fix or remove the credit ledger file named above, then call again".into(),
            })?;
        check_limits(cost, el.per_call_credit_limit, el.daily_credit_limit, today).map_err(|e| {
            SfxError::CreditLimit {
                message: e.to_string(),
                fix: e.fix(),
            }
        })
    }

    async fn finish(&self, snap: JobSnapshot, wait: bool) -> Result<CallToolResult, McpError> {
        if !wait {
            return Ok(structured(JobStarted {
                job_id: snap.job_id,
                status: snap.status,
                lane: snap.lane,
                progress: snap.progress_text,
                hint: "poll get_job with this job_id; cancel_job stops it".into(),
            }));
        }
        // If the call itself is cancelled (tasks/cancel, request cancellation), stop the job too.
        let guard = CancelOnDrop::new(self.state.jobs.clone(), snap.job_id.clone());
        let done = self.state.jobs.wait_with_progress(&snap.job_id).await?;
        guard.disarm();
        match (done.status, done.result, done.error) {
            (JobStatus::Done, Some(result), _) => generate_result(result),
            (JobStatus::Cancelled, ..) => Err(SfxError::Cancelled.into()),
            (_, _, Some(error)) => job_failure(&error),
            _ => Err(McpError::internal_error(
                "the job ended without a result",
                None,
            )),
        }
    }
}

/// A finished job as the tool result: lean structured content, a text summary, and a `resource_link` per file.
fn generate_result(value: Value) -> Result<CallToolResult, McpError> {
    let result: GenerateResult = serde_json::from_value(value.clone())
        .map_err(|e| McpError::internal_error(format!("malformed job result: {e}"), None))?;
    let mut out = CallToolResult::structured(value);
    out.content = vec![ContentBlock::text(summary(&result))];
    out.content.extend(
        result
            .files
            .iter()
            .map(|f| ContentBlock::resource_link(file_resource(f))),
    );
    Ok(out)
}

fn summary(result: &GenerateResult) -> String {
    let mut text = format!(
        "Made {} sound{} with {} ({}) in {:.1} s",
        result.files.len(),
        if result.files.len() == 1 { "" } else { "s" },
        result.model,
        result.runs,
        result.timings.get("total").copied().unwrap_or(0.0)
    );
    if let Some(credits) = result.credits {
        text.push_str(&format!(", {credits} credits"));
    }
    text.push_str(":\n");
    for f in &result.files {
        text.push_str(&format!(
            "  {}  {:.2} s, {} Hz {}, {}, {:.1} dBTP{}\n",
            f.path,
            f.duration_s,
            f.sample_rate,
            if f.channels == 1 { "mono" } else { "stereo" },
            f.lufs
                .map_or("no integrated loudness".to_string(), |l| format!(
                    "{l:.1} LUFS"
                )),
            f.true_peak_dbtp,
            f.seed.map_or(String::new(), |s| format!(", seed {s}")),
        ));
    }
    text.push_str("Raw copies (`.raw.wav`) and provenance (`.json`) sit next to each file.");
    for w in &result.warnings {
        text.push_str(&format!("\nWarning: {w}"));
    }
    text
}

fn file_resource(file: &FileResult) -> Resource {
    let name = std::path::Path::new(&file.path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| file.path.clone());
    let mut resource = Resource::new(file_uri(&file.path), name)
        .with_description(format!("{:.2} s processed WAV", file.duration_s))
        .with_mime_type("audio/wav");
    if let Ok(meta) = std::fs::metadata(&file.path) {
        resource = resource.with_size(meta.len());
    }
    resource
}

/// `file://` URI for an absolute path (percent-encoding everything but unreserved characters and `/`).
fn file_uri(path: &str) -> String {
    let path = path.replace('\\', "/");
    let mut out = String::from("file://");
    if !path.starts_with('/') {
        out.push('/');
    }
    for b in path.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' | b':' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// A failed job as the tool answers it: caller mistakes as `isError` results, the rest as JSON-RPC errors (§7).
fn job_failure(error: &crate::jobs::JobError) -> Result<CallToolResult, McpError> {
    if error.caller_error {
        return Ok(CallToolResult::structured_error(
            serde_json::to_value(ErrorBody {
                error: ErrorDetail {
                    code: error.code.clone(),
                    message: error.message.clone(),
                    fix: error.fix.clone(),
                },
            })
            .unwrap_or(Value::Null),
        ));
    }
    let mut data = serde_json::json!({ "code": error.code, "fix": error.fix });
    if let Some(stage) = &error.stage {
        data["stage"] = stage.clone().into();
    }
    if let Some(kind) = &error.kind {
        data["kind"] = kind.clone().into();
    }
    Err(McpError::internal_error(error.message.clone(), Some(data)))
}

fn structured(value: impl Serialize) -> CallToolResult {
    CallToolResult::structured(serde_json::to_value(value).unwrap_or(Value::Null))
}

#[tool_router]
impl SfxGenServer {
    #[tool(
        description = "Generate sound effects from a text prompt. Writes processed WAV files (true-peak limited, default -1 dBTP) \
with the untouched raw copy and a provenance JSON beside each, named <name>_01, <name>_02, ... for up to 4 variations. Local \
Stable Audio 3 takes about 1.5 s warm; a cloud model (ElevenLabs) costs credits and sends the prompt to its provider, and is never \
chosen for you. Options: duration_s, seed (local), loop (seamless), name/output_dir/output_path, post {trim_silence, fades, \
loudness_lufs, true_peak_dbtp, mono, sample_rate, bit_depth}. Pass wait=false to get a job_id and poll get_job. Mistakes in the \
call come back as isError results with a fix.",
        annotations(open_world_hint = true),
        output_schema = rmcp::handler::server::tool::schema_for_output::<GenerateOutput>()
    )]
    async fn generate_sfx(
        &self,
        Parameters(input): Parameters<SfxRequest>,
    ) -> Result<CallToolResult, McpError> {
        let (snap, resolved) = match self.submit(&input).await {
            Ok(pair) => pair,
            Err(e) => return e.into_tool_result(),
        };
        self.finish(snap, resolved.wait).await
    }

    #[tool(
        description = "Get status, stage progress and result of a sound-effect job (from generate_sfx with wait=false).",
        output_schema = rmcp::handler::server::tool::schema_for_output::<JobOutput>()
    )]
    async fn get_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        match self.state.jobs.get(&input.job_id) {
            Ok(snap) => Ok(structured(snap)),
            Err(e) => e.into_tool_result(),
        }
    }

    #[tool(
        description = "Cancel a queued or running sound-effect job. A local worker is stopped and no partial files are left behind.",
        output_schema = rmcp::handler::server::tool::schema_for_output::<JobOutput>()
    )]
    async fn cancel_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        match self.state.jobs.cancel(&input.job_id) {
            Ok(snap) => Ok(structured(snap)),
            Err(e) => e.into_tool_result(),
        }
    }

    #[tool(
        description = "List sound-effect models: status (ready / missing_files with sizes and the exact download command / \
needs_licence_acceptance with what to accept / needs_credentials / backend_unavailable with why), licences and whether \
commercial use is allowed, where the prompt goes (cloud), memory needed, typical time and duration range. Never loads weights or \
spends credits.",
        output_schema = rmcp::handler::server::tool::schema_for_output::<ListModelsOutput>()
    )]
    async fn list_models(&self) -> Result<CallToolResult, McpError> {
        let st = &self.state;
        let mut models = Vec::new();
        for model in st.registry.models() {
            models.push(self.model_entry(model).await);
        }
        let path = |p: &std::path::Path| p.display().to_string();
        Ok(structured(ListModelsOutput {
            models,
            default_model: st.cfg.default_model.clone(),
            available_memory_mb: st.memory.available_mb(),
            allowed_roots: st.cfg.allowed_roots.iter().map(|r| path(r)).collect(),
            default_output_dir: path(&st.cfg.default_output_dir),
            config_dir: path(&st.cfg.config_dir),
            hf_cache: path(&st.cfg.hf_cache),
            gpu_lock_holder: st
                .jobs
                .deps()
                .gpu_lock
                .current_holder()
                .map(|h| h.describe()),
            jobs_busy: st.jobs.busy(),
        }))
    }

    #[tool(
        description = "ElevenLabs plan and credits, when the API key can read the account (a key without the user_read permission \
can't: readable is false with the reason), plus the local credit ledger's totals for today and this month and the configured \
limits. Free; reads no audio.",
        annotations(open_world_hint = true),
        output_schema = rmcp::handler::server::tool::schema_for_output::<AccountStatusOutput>()
    )]
    async fn account_status(&self) -> Result<CallToolResult, McpError> {
        let st = &self.state;
        let el = &st.cfg.elevenlabs;
        let totals = st.ledger.totals().map_err(|e| {
            McpError::from(SfxError::Io {
                path: st.ledger.path().to_path_buf(),
                message: e.to_string(),
            })
        })?;
        let credentials_present = self.credentials_present().await;
        let (readable, reason, account) = match st.jobs.deps().backends.get("elevenlabs") {
            None => (
                false,
                Some("the elevenlabs backend isn't in this build".to_string()),
                None,
            ),
            Some(backend) => match backend.account().await {
                Some(Ok(snapshot)) => (true, None, Some(snapshot)),
                Some(Err(AccountError::NotReadable { reason })) => (false, Some(reason), None),
                Some(Err(AccountError::Failed(failure))) => (
                    false,
                    Some(secrets::scrub(
                        &failure.into_error("elevenlabs").to_string(),
                    )),
                    None,
                ),
                None => (false, Some("this backend has no account".to_string()), None),
            },
        };
        Ok(structured(AccountStatusOutput {
            provider: "elevenlabs".into(),
            configured_plan: el.plan.clone(),
            credentials_present,
            readable,
            reason,
            account,
            ledger: LedgerTotals {
                today: totals.today,
                month: totals.month,
                per_call_limit: el.per_call_credit_limit,
                daily_limit: el.daily_credit_limit,
                credits_per_second: el.credits_per_second,
                credits_auto_duration: el.credits_auto_duration,
            },
        }))
    }

    #[tool(
        description = "Stop the kept-alive local model worker to free its memory (it also unloads itself after idle_unload_s). \
Refused while local jobs are queued or running.",
        output_schema = rmcp::handler::server::tool::schema_for_output::<UnloadOutput>()
    )]
    async fn unload_models(&self) -> Result<CallToolResult, McpError> {
        match self.state.jobs.unload_all().await {
            Ok(unloaded) => Ok(structured(UnloadOutput {
                unloaded,
                available_memory_mb: self.state.memory.available_mb(),
            })),
            Err(e) => e.into_tool_result(),
        }
    }
}

adk_mcp_sdk::mcp_2026_server! {
    server: SfxGenServer,
    task_tools: ["generate_sfx"],
    task_ttl_overrides: [("generate_sfx", TASK_TTL_MS)],
    approval_tools: [],
    mutating_tools: ["generate_sfx", "cancel_job", "unload_models"],
    destructive_tools: [],
    idempotent_tools: ["cancel_job", "unload_models"],
    cache_ttl_ms: 60_000,
    instructions: "Sound effects from text. Call list_models first to see which models are ready, what their licences allow \
and where the prompt goes. generate_sfx writes processed WAV files (limited to -1 dBTP by default) with a raw copy and a \
provenance JSON each; a cloud model costs credits and is never picked for you, and a call over the credit limits is refused \
before anything is sent. Mistakes in a call come back as isError results with a fix. Pass wait=false and poll get_job if your \
client times out; account_status shows credits spent.",
}

#[async_trait::async_trait]
impl HealthCheck for SfxGenServer {
    async fn check_health(&self) -> HealthStatus {
        let started = std::time::Instant::now();
        let backends = &self.state.jobs.deps().backends;
        let mut parts = Vec::new();
        for model in self.state.registry.models() {
            let state = match backends.availability_of(model).await {
                // Backend only: licence and credentials are per-model statuses in list_models.
                Availability::Ready => "backend available".to_string(),
                Availability::Unavailable { reason } => format!("backend unavailable ({reason})"),
            };
            parts.push(format!("{}: {state}", model.id));
        }
        HealthStatus {
            // A backend that can't run here is reported per model by list_models; the server itself is fine.
            healthy: true,
            message: Some(format!(
                "{} model(s); {}",
                self.state.registry.models().len(),
                if parts.is_empty() {
                    "none".to_string()
                } else {
                    parts.join(", ")
                }
            )),
            latency_ms: Some(started.elapsed().as_millis() as u64),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::{MOCK_MODELS, MockBackend};
    use crate::heavy::FixedProcesses;
    use crate::memory::FixedMemory;

    fn server_with(
        root: &std::path::Path,
        config: &str,
        models: &str,
        backends: BackendRegistry,
    ) -> SfxGenServer {
        let dir = root.join(".sfxgen");
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = Config::from_toml(&dir, config).unwrap();
        // Never touch the real ledger, cache or GPU lock from a test.
        cfg.cache_root = root.join("cache");
        cfg.data_root = root.join("data");
        let registry = ModelRegistry::from_toml(models).unwrap();
        SfxGenServer::new(
            cfg,
            registry,
            backends,
            Arc::new(FixedMemory(8000)),
            Arc::new(FixedProcesses(vec![])),
        )
    }

    fn server(root: &std::path::Path, models: &str, backends: BackendRegistry) -> SfxGenServer {
        server_with(root, "", models, backends)
    }

    #[tokio::test]
    async fn list_models_reports_statuses_licences_and_privacy() {
        let root = tempfile::tempdir().unwrap();
        let s = server(
            root.path(),
            crate::config::MODELS_EXAMPLE,
            BackendRegistry::default(),
        );
        let result = s.list_models().await.unwrap();
        let body = result.structured_content.unwrap();
        let models = body["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        let sa3 = &models[0];
        assert_eq!(sa3["id"], "sa3-small-sfx");
        assert_eq!(sa3["runs"], "local");
        assert_eq!(sa3["loaded"], false);
        assert_eq!(sa3["memory_mb"], 3700);
        assert_eq!(sa3["needs_mb_free"], 4700);
        assert_eq!(sa3["typical_s"], 1.5);
        assert_eq!(sa3["commercial_use"], "conditional");
        assert_eq!(sa3["licences"].as_array().unwrap().len(), 3);
        // No backend registered, so the reason leads.
        assert_eq!(sa3["status"], "backend_unavailable");
        assert!(sa3["detail"].as_str().unwrap().contains("sa3_mlx"));
        assert!(sa3["reason"].as_str().unwrap().contains("sa3_mlx"));
        let el = &models[1];
        assert_eq!(el["runs"], "cloud");
        assert_eq!(el["privacy"], "the prompt is sent to ElevenLabs");
        assert_eq!(
            el["commercial_use"], "no",
            "the Free plan is non-commercial"
        );
        assert!(el["commercial_note"].as_str().unwrap().contains("free"));
        assert!(el.get("needs_mb_free").is_none());
        assert!(el["credentials_present"].is_boolean());
        assert_eq!(body["default_model"], "sa3-small-sfx");
        assert_eq!(body["available_memory_mb"], 8000);
        assert_eq!(body["jobs_busy"], 0);
    }

    #[tokio::test]
    async fn ready_mock_models_list_as_ready() {
        let root = tempfile::tempdir().unwrap();
        let mut backends = BackendRegistry::default();
        backends.insert(Arc::new(MockBackend::local(Duration::from_millis(1))));
        backends.insert(Arc::new(MockBackend::cloud(Duration::from_millis(1))));
        let s = server(root.path(), MOCK_MODELS, backends);
        let body = s.list_models().await.unwrap().structured_content.unwrap();
        let models = body["models"].as_array().unwrap();
        assert!(models.iter().all(|m| m["status"] == "ready"), "{models:?}");
        assert!(models[0].get("detail").is_none());
        assert_eq!(models[1]["commercial_use"], "conditional");
        let health = s.check_health().await;
        assert!(health.healthy);
        assert!(
            health
                .message
                .unwrap()
                .contains("mock-local: backend available")
        );
    }

    fn mock_server(root: &std::path::Path, config: &str) -> (SfxGenServer, Arc<MockBackend>) {
        let cloud = Arc::new(MockBackend::cloud(Duration::from_millis(1)));
        let mut backends = BackendRegistry::default();
        backends.insert(Arc::new(MockBackend::local(Duration::from_millis(1))));
        backends.insert(cloud.clone());
        let config = format!(
            "allowed_roots = [\"game\"]\ndefault_output_dir = \"game/sfx\"\ndefault_model = \"mock-local\"\n{config}"
        );
        std::fs::create_dir_all(root.join("game")).unwrap();
        (server_with(root, &config, MOCK_MODELS, backends), cloud)
    }

    fn input(json: serde_json::Value) -> Parameters<SfxRequest> {
        Parameters(serde_json::from_value(json).unwrap())
    }

    #[tokio::test]
    async fn generate_writes_files_and_answers_with_links_a_summary_and_structured_content() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        let out = s
            .generate_sfx(input(
                serde_json::json!({"prompt": "a meep", "variations": 2, "duration_s": 1.0}),
            ))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(false));
        let v = out.structured_content.as_ref().unwrap();
        assert_eq!(v["files"].as_array().unwrap().len(), 2);
        assert_eq!(v["model"], "mock-local");
        // text summary + one resource_link per file
        assert!(
            out.content[0]
                .as_text()
                .unwrap()
                .text
                .starts_with("Made 2 sounds with mock-local (local)")
        );
        let links: Vec<_> = out
            .content
            .iter()
            .filter_map(|c| c.as_resource_link())
            .collect();
        assert_eq!(links.len(), 2);
        assert!(
            links[0].uri.starts_with("file:///") && links[0].uri.ends_with("a-meep_01.wav"),
            "{}",
            links[0].uri
        );
        assert_eq!(links[0].mime_type.as_deref(), Some("audio/wav"));
        assert!(links[0].size.unwrap() > 1000);
        for f in v["files"].as_array().unwrap() {
            assert!(std::path::Path::new(f["path"].as_str().unwrap()).exists());
            assert!(std::path::Path::new(f["sidecar"].as_str().unwrap()).exists());
        }
    }

    #[tokio::test]
    async fn caller_errors_are_is_error_results_with_a_fix() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        for (args, code) in [
            (serde_json::json!({"prompt": ""}), "invalid_params"),
            (
                serde_json::json!({"prompt": "x", "variations": 9}),
                "invalid_params",
            ),
            (
                serde_json::json!({"prompt": "x", "duration_s": 99.0}),
                "duration_out_of_range",
            ),
            (
                serde_json::json!({"prompt": "x", "model": "nope"}),
                "unknown_model",
            ),
            (
                serde_json::json!({"prompt": "x", "loop": true, "post": {"trim_silence": true}}),
                "option_conflict",
            ),
            (
                serde_json::json!({"prompt": "x", "output_dir": "/etc"}),
                "path_outside_roots",
            ),
        ] {
            let out = s.generate_sfx(input(args.clone())).await.unwrap();
            assert_eq!(out.is_error, Some(true), "{args}");
            let body: ErrorBody = serde_json::from_value(out.structured_content.unwrap()).unwrap();
            assert_eq!(body.error.code, code, "{args}");
            assert!(!body.error.fix.is_empty());
        }
        let out = s
            .get_job(Parameters(JobIdInput {
                job_id: "nope".into(),
            }))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(true));
        assert_eq!(
            out.structured_content.unwrap()["error"]["code"],
            "unknown_job"
        );
    }

    #[tokio::test]
    async fn a_taken_exact_path_is_refused_up_front() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        let args =
            serde_json::json!({"prompt": "x", "output_path": "game/one.wav", "duration_s": 1.0});
        s.generate_sfx(input(args.clone())).await.unwrap();
        let out = s.generate_sfx(input(args)).await.unwrap();
        assert_eq!(out.is_error, Some(true));
        assert_eq!(
            out.structured_content.unwrap()["error"]["code"],
            "output_exists"
        );
    }

    #[tokio::test]
    async fn the_credit_guard_refuses_before_anything_is_sent() {
        let root = tempfile::tempdir().unwrap();
        let (s, cloud) = mock_server(
            root.path(),
            "[elevenlabs]\nper_call_credit_limit = 25\ndaily_credit_limit = 40\n",
        );
        // 2 x 3 s = 60 credits: over the per-call limit.
        let out = s
            .generate_sfx(input(serde_json::json!({"prompt": "x", "model": "mock-cloud", "variations": 2, "duration_s": 3.0})))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(true));
        let body = out.structured_content.unwrap();
        assert_eq!(body["error"]["code"], "credit_limit");
        assert!(
            body["error"]["fix"]
                .as_str()
                .unwrap()
                .contains("per_call_credit_limit")
        );
        assert_eq!(
            cloud.generations.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
        assert_eq!(cloud.active.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(!root.path().join("game/sfx").exists(), "nothing reserved");
        // Under the per-call limit it runs (20 credits) and lands in the ledger.
        let out = s
            .generate_sfx(input(
                serde_json::json!({"prompt": "x", "model": "mock-cloud", "duration_s": 2.0}),
            ))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(false), "{out:?}");
        assert_eq!(out.structured_content.as_ref().unwrap()["credits"], 20);
        // Another 20 would reach 40 (allowed); then 10 more passes the daily limit.
        let out = s
            .generate_sfx(input(
                serde_json::json!({"prompt": "y", "model": "mock-cloud", "duration_s": 2.0}),
            ))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(false));
        let out = s
            .generate_sfx(input(
                serde_json::json!({"prompt": "z", "model": "mock-cloud", "duration_s": 1.0}),
            ))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(true));
        let body = out.structured_content.unwrap();
        assert_eq!(body["error"]["code"], "credit_limit");
        assert!(
            body["error"]["message"].as_str().unwrap().contains("daily"),
            "{body}"
        );
        assert_eq!(
            cloud.generations.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
        // The ledger carries the spend.
        let status = s
            .account_status()
            .await
            .unwrap()
            .structured_content
            .unwrap();
        assert_eq!(status["ledger"]["today"], 40);
        assert_eq!(status["ledger"]["daily_limit"], 40);
    }

    #[tokio::test]
    async fn a_model_that_isnt_ready_is_refused_with_alternatives_and_never_swapped() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        // A registry with a local model whose backend is missing and a ready cloud one.
        let models = format!(
            "{MOCK_MODELS}\n[[model]]\nid = \"ghost\"\nbackend = \"mock-gone\"\nduration_s = {{ min = 0.5, max = 5.0 }}\nmemory_mb = 10\ntypical_s = 1\noptions = {{ kind = \"mock\", runs = \"local\" }}\nlicence = [{{ component = \"x\", name = \"MIT\", commercial = \"yes\" }}]\n"
        );
        drop(s);
        let mut backends = BackendRegistry::default();
        backends.insert(Arc::new(MockBackend::local(Duration::from_millis(1))));
        backends.insert(Arc::new(MockBackend::cloud(Duration::from_millis(1))));
        let s = server_with(
            root.path(),
            "allowed_roots = [\"game\"]\ndefault_model = \"ghost\"\n",
            &models,
            backends,
        );
        let out = s
            .generate_sfx(input(serde_json::json!({"prompt": "x"})))
            .await
            .unwrap();
        assert_eq!(out.is_error, Some(true));
        let body: ErrorBody = serde_json::from_value(out.structured_content.unwrap()).unwrap();
        assert_eq!(body.error.code, "model_not_ready");
        assert!(
            body.error.message.contains("mock-local") && body.error.message.contains("mock-cloud"),
            "{}",
            body.error.message
        );
        assert!(body.error.message.contains("backend_unavailable"));
    }

    #[tokio::test]
    async fn wait_false_returns_a_job_id_and_get_job_follows_it() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        let out = s
            .generate_sfx(input(
                serde_json::json!({"prompt": "x", "wait": false, "duration_s": 1.0}),
            ))
            .await
            .unwrap();
        let started = out.structured_content.unwrap();
        let id = started["job_id"].as_str().unwrap().to_string();
        assert!(started["hint"].as_str().unwrap().contains("get_job"));
        let snap = loop {
            let r = s
                .get_job(Parameters(JobIdInput { job_id: id.clone() }))
                .await
                .unwrap();
            let v = r.structured_content.unwrap();
            if v["status"] == "done" {
                break v;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        assert_eq!(snap["result"]["files"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn account_status_without_the_backend_says_why() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        let v = s
            .account_status()
            .await
            .unwrap()
            .structured_content
            .unwrap();
        assert_eq!(v["readable"], false);
        assert!(
            v["reason"]
                .as_str()
                .unwrap()
                .contains("isn't in this build")
        );
        assert_eq!(v["configured_plan"], "free");
        assert_eq!(v["ledger"]["today"], 0);
        assert!(v["credentials_present"].is_boolean());
    }

    #[test]
    fn file_uris_are_percent_encoded() {
        assert_eq!(file_uri("/a b/c#d.wav"), "file:///a%20b/c%23d.wav");
        assert_eq!(file_uri("C:\\x\\y.wav"), "file:///C:/x/y.wav");
        assert_eq!(file_uri("/tmp/ä.wav"), "file:///tmp/%C3%A4.wav");
    }

    #[tokio::test]
    async fn unload_models_reports_what_was_unloaded() {
        let root = tempfile::tempdir().unwrap();
        let (s, _) = mock_server(root.path(), "");
        s.generate_sfx(input(serde_json::json!({"prompt": "x", "duration_s": 1.0})))
            .await
            .unwrap();
        let v = s.unload_models().await.unwrap().structured_content.unwrap();
        assert_eq!(v["unloaded"], serde_json::json!(["mock-local"]));
        let v = s.unload_models().await.unwrap().structured_content.unwrap();
        assert_eq!(v["unloaded"], serde_json::json!([]));
    }
}
