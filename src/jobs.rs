//! Job manager (design §8; R4, R11).
//!
//! Two lanes, each with a bounded FIFO queue (`queue_depth`) and one worker: local backends run strictly one job at a
//! time, and so do cloud backends, but a cloud job never waits behind a local one (a cloud backend may run its
//! variations concurrently inside one job). Every job has a `watch` channel carrying its latest snapshot, so callers
//! can wait, poll or stream progress. Finished jobs are kept for `job_ttl`; the local lane arms an idle unload.
//!
//! The local lane, per job: take the cross-process GPU lock (holder `mcp-sfxgen`), unload any other resident local
//! backend, pass the in-job memory preflight (unless the model is already resident in a kept worker), run the
//! backend, and release the lock as soon as the backend returns, so post-processing never blocks a sister server. The
//! cloud lane takes no lock and runs no preflight.
//!
//! The backend returns raw clips; what happens next is the job's `JobFinisher` (the server's pipeline: loop, edits,
//! limiter, naming, sidecar, provenance, task 12). It reserves names in `prepare`, turns clips into files and a final
//! `serde_json::Value` in `finish`, and cleans up in `abort`; the manager stores the value as the job's result.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::backend::{
    BackendError, BackendRegistry, BackendRequest, ProgressSink, RawClip, Runs, Stage,
};
use crate::error::SfxError;
use crate::gpu_lock::{GpuLock, Holder};
use crate::heavy::{ProcessLister, advice, census};
use crate::memory::{MemoryProbe, preflight};
use crate::registry::ModelSpec;

/// What the job manager needs from the rest of the server.
#[derive(Clone)]
pub struct JobDeps {
    pub backends: BackendRegistry,
    pub gpu_lock: GpuLock,
    pub memory: Arc<dyn MemoryProbe>,
    pub processes: Arc<dyn ProcessLister>,
    pub heavy_processes: Vec<String>,
    pub memory_headroom_mb: u64,
    /// Jobs waiting per lane.
    pub queue_depth: usize,
    pub job_ttl: Duration,
    /// Unload local backends after this long without local work; zero = never (only `unload_models`).
    pub idle_unload: Duration,
    /// Scratch folders are made under here, one per job (`BackendRequest::out_dir`), and removed afterwards.
    pub scratch_dir: PathBuf,
}

/// Which queue a job waits in, from its backend's `runs()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    Local,
    Cloud,
}

impl Lane {
    const ALL: [Lane; 2] = [Lane::Local, Lane::Cloud];

    fn index(self) -> usize {
        match self {
            Self::Local => 0,
            Self::Cloud => 1,
        }
    }
}

impl From<Runs> for Lane {
    fn from(runs: Runs) -> Self {
        match runs {
            Runs::Local => Self::Local,
            Runs::Cloud => Self::Cloud,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

/// What the job manager hands the finisher besides the clips.
pub struct FinishContext<'a> {
    pub job_id: &'a str,
    pub model: &'a ModelSpec,
    /// The request as the backend saw it (`out_dir` is the job's scratch folder).
    pub request: &'a BackendRequest,
    pub backend_id: &'a str,
    pub runs: Runs,
    /// Who got the prompt (cloud backends).
    pub provider: Option<&'static str>,
    /// Seconds so far: `queued`, `gpu_lock_wait` (local), `generate`. The finisher adds its own to the result.
    pub timings: BTreeMap<String, f64>,
    /// Report `Stage::PostProcessing` / `Stage::Writing` here.
    pub progress: &'a ProgressSink,
    pub cancel: &'a CancellationToken,
}

/// The server's post-backend pipeline for one job (task 12). Calls arrive in order: `prepare`, then the backend runs,
/// then `finish` (backend succeeded) or `abort` (anything failed or was cancelled after `prepare` succeeded).
#[async_trait::async_trait]
pub trait JobFinisher: Send + Sync {
    /// Before the backend runs, after the GPU lock and the preflight: reserve file names. A failure fails the job.
    async fn prepare(&self) -> Result<(), SfxError> {
        Ok(())
    }

    /// Turn the raw clips into files, and return the job's result. Long CPU work belongs in `spawn_blocking`.
    async fn finish(&self, clips: Vec<RawClip>, ctx: &FinishContext<'_>)
    -> Result<Value, SfxError>;

    /// Failure or cancel: delete partial files and reservations. Idempotent.
    async fn abort(&self) {}
}

/// A finisher that writes nothing and returns a summary of the clips: for the mock server and tests.
pub struct SummaryFinisher;

#[async_trait::async_trait]
impl JobFinisher for SummaryFinisher {
    async fn finish(
        &self,
        clips: Vec<RawClip>,
        ctx: &FinishContext<'_>,
    ) -> Result<Value, SfxError> {
        Ok(json!({
            "model": ctx.model.id,
            "backend": ctx.backend_id,
            "runs": ctx.runs,
            "clips": clips.iter().map(|c| json!({
                "seed": c.seed,
                "duration_s": c.duration_s(),
                "sample_rate": c.sample_rate,
                "channels": c.channels,
                "credits": c.credits,
            })).collect::<Vec<_>>(),
            "timings": ctx.timings,
        }))
    }
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct JobError {
    pub code: String,
    pub message: String,
    /// A caller mistake (answered as an `isError` result) rather than a server or backend fault.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub caller_error: bool,
    /// The next step, when the error has one.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub fix: String,
    /// For backend failures: the stage and kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
pub struct JobSnapshot {
    pub job_id: String,
    pub status: JobStatus,
    pub lane: Lane,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Stage>,
    /// Human-readable progress, e.g. "generating 2 of 4".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_text: Option<String>,
    /// What the job's finisher returned (a `GenerateResult` for `generate_sfx` jobs).
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<crate::result::GenerateResult>")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
}

/// A validated job ready to run.
#[derive(Clone)]
pub struct JobRequest {
    pub model: ModelSpec,
    /// `out_dir` is filled in by the manager.
    pub request: BackendRequest,
    pub finisher: Arc<dyn JobFinisher>,
}

struct JobEntry {
    tx: watch::Sender<JobSnapshot>,
    lane: Lane,
    cancel: CancellationToken,
    request: Mutex<Option<JobRequest>>,
    submitted_at: Instant,
    finished_at: Mutex<Option<Instant>>,
}

impl JobEntry {
    fn snapshot(&self) -> JobSnapshot {
        self.tx.borrow().clone()
    }

    fn set_stage(&self, stage: Stage) {
        self.tx.send_modify(|s| {
            s.progress_text = Some(stage.to_string());
            s.progress = Some(stage);
        });
    }

    fn finish(&self, status: JobStatus, result: Option<Value>, error: Option<JobError>) {
        self.tx.send_modify(|s| {
            s.status = status;
            s.result = result;
            s.error = error;
            s.progress = None;
            s.progress_text = None;
        });
        *self.finished_at.lock().unwrap() = Some(Instant::now());
    }

    fn fail(&self, err: &SfxError) {
        if matches!(err, SfxError::Cancelled) {
            self.finish(JobStatus::Cancelled, None, None);
        } else {
            self.finish(JobStatus::Failed, None, Some(to_job_error(err)));
        }
    }
}

#[derive(Default)]
struct LaneState {
    queue: Mutex<VecDeque<String>>,
    running: Mutex<Option<String>>,
    wake: Notify,
}

struct Inner {
    jobs: Mutex<HashMap<String, Arc<JobEntry>>>,
    lanes: [LaneState; 2],
    deps: JobDeps,
}

#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Inner>,
}

impl JobManager {
    /// Create the manager and spawn one worker per lane on the current tokio runtime.
    pub fn new(deps: JobDeps) -> Self {
        let inner = Arc::new(Inner {
            jobs: Mutex::new(HashMap::new()),
            lanes: Default::default(),
            deps,
        });
        for lane in Lane::ALL {
            tokio::spawn(worker(Arc::downgrade(&inner), lane));
        }
        Self { inner }
    }

    pub fn deps(&self) -> &JobDeps {
        &self.inner.deps
    }

    /// The lane a model's backend runs in (`Local` when the backend isn't registered; the job then fails on its own).
    pub fn lane_of(&self, model: &ModelSpec) -> Lane {
        self.inner
            .deps
            .backends
            .get(&model.backend)
            .map(|b| Lane::from(b.runs()))
            .unwrap_or(Lane::Local)
    }

    /// Queue a job in its backend's lane. Fails with `QueueFull` when `queue_depth` jobs are already waiting there.
    pub fn submit(&self, job: JobRequest) -> Result<JobSnapshot, SfxError> {
        self.prune();
        let id = new_job_id();
        let lane = self.lane_of(&job.model);
        let mut queue = self.inner.lane(lane).queue.lock().unwrap();
        let max = self.inner.deps.queue_depth;
        if queue.len() >= max {
            return Err(SfxError::QueueFull { max });
        }
        let stage = Stage::Queued {
            position: queue.len() + 1,
        };
        let snapshot = JobSnapshot {
            job_id: id.clone(),
            status: JobStatus::Queued,
            lane,
            progress_text: Some(stage.to_string()),
            progress: Some(stage),
            result: None,
            error: None,
        };
        let (tx, _) = watch::channel(snapshot.clone());
        let entry = Arc::new(JobEntry {
            tx,
            lane,
            cancel: CancellationToken::new(),
            request: Mutex::new(Some(job)),
            submitted_at: Instant::now(),
            finished_at: Mutex::new(None),
        });
        self.inner.jobs.lock().unwrap().insert(id.clone(), entry);
        queue.push_back(id);
        drop(queue);
        self.inner.lane(lane).wake.notify_one();
        Ok(snapshot)
    }

    pub fn get(&self, id: &str) -> Result<JobSnapshot, SfxError> {
        self.prune();
        Ok(self.entry(id)?.snapshot())
    }

    pub fn subscribe(&self, id: &str) -> Result<watch::Receiver<JobSnapshot>, SfxError> {
        Ok(self.entry(id)?.tx.subscribe())
    }

    /// Wait until the job reaches a terminal state.
    pub async fn wait(&self, id: &str) -> Result<JobSnapshot, SfxError> {
        let mut rx = self.subscribe(id)?;
        loop {
            let snap = rx.borrow_and_update().clone();
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(SfxError::UnknownJob(id.to_string()));
            }
        }
    }

    /// Wait for a job, mirroring each new progress text into the MCP task status (when the call runs as a Task).
    pub async fn wait_with_progress(&self, id: &str) -> Result<JobSnapshot, SfxError> {
        let mut rx = self.subscribe(id)?;
        let mut last = String::new();
        loop {
            let snap = rx.borrow_and_update().clone();
            if let Some(text) = &snap.progress_text
                && *text != last
            {
                adk_mcp_sdk::set_current_task_status(text.clone());
                last = text.clone();
            }
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(SfxError::UnknownJob(id.to_string()));
            }
        }
    }

    /// Cancel a queued job immediately, or signal a running one. Idempotent on finished jobs.
    pub fn cancel(&self, id: &str) -> Result<JobSnapshot, SfxError> {
        let entry = self.entry(id)?;
        let removed = {
            let mut queue = self.inner.lane(entry.lane).queue.lock().unwrap();
            let before = queue.len();
            queue.retain(|q| q != id);
            before != queue.len()
        };
        entry.cancel.cancel();
        if removed {
            entry.request.lock().unwrap().take();
            entry.finish(JobStatus::Cancelled, None, None);
            self.inner.renumber_queue(entry.lane);
        }
        Ok(entry.snapshot())
    }

    /// Cancel every queued and running job (server shutdown).
    pub fn cancel_all(&self) {
        let ids: Vec<String> = self.inner.jobs.lock().unwrap().keys().cloned().collect();
        for id in ids {
            let _ = self.cancel(&id);
        }
    }

    /// Jobs queued plus running, across both lanes.
    pub fn busy(&self) -> usize {
        Lane::ALL.iter().map(|l| self.busy_in(*l)).sum()
    }

    /// Jobs queued plus the one running in `lane`.
    pub fn busy_in(&self, lane: Lane) -> usize {
        let state = self.inner.lane(lane);
        state.queue.lock().unwrap().len() + usize::from(state.running.lock().unwrap().is_some())
    }

    /// Unload every backend now. Refused while a local job is running or queued; cloud jobs hold no local memory, so
    /// they don't block it.
    pub async fn unload_all(&self) -> Result<Vec<String>, SfxError> {
        let local = self.busy_in(Lane::Local);
        if local > 0 {
            return Err(SfxError::invalid(
                "unload_models",
                format!("{local} local job(s) queued or running; cancel them first or wait"),
            ));
        }
        Ok(self.inner.unload_all().await)
    }

    fn entry(&self, id: &str) -> Result<Arc<JobEntry>, SfxError> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| SfxError::UnknownJob(id.to_string()))
    }

    fn prune(&self) {
        let ttl = self.inner.deps.job_ttl;
        self.inner.jobs.lock().unwrap().retain(|_, e| {
            e.finished_at
                .lock()
                .unwrap()
                .is_none_or(|t| t.elapsed() < ttl)
        });
    }
}

impl Inner {
    fn lane(&self, lane: Lane) -> &LaneState {
        &self.lanes[lane.index()]
    }

    fn renumber_queue(&self, lane: Lane) {
        let queue = self.lane(lane).queue.lock().unwrap();
        let jobs = self.jobs.lock().unwrap();
        for (i, id) in queue.iter().enumerate() {
            if let Some(e) = jobs.get(id) {
                e.set_stage(Stage::Queued { position: i + 1 });
            }
        }
    }

    async fn unload_all(&self) -> Vec<String> {
        let mut unloaded = Vec::new();
        for backend in self.deps.backends.all() {
            if let Some(model) = backend.loaded_model() {
                backend.unload().await;
                unloaded.push(model);
            }
        }
        unloaded
    }

    fn next_job(&self, lane: Lane) -> Option<(String, Arc<JobEntry>)> {
        let state = self.lane(lane);
        let id = state.queue.lock().unwrap().pop_front()?;
        let entry = self.jobs.lock().unwrap().get(&id).cloned()?;
        *state.running.lock().unwrap() = Some(id.clone());
        self.renumber_queue(lane);
        Some((id, entry))
    }

    async fn run(&self, id: &str, entry: &JobEntry) {
        let Some(job) = entry.request.lock().unwrap().take() else {
            return;
        };
        if entry.cancel.is_cancelled() {
            entry.finish(JobStatus::Cancelled, None, None);
            return;
        }
        entry.tx.send_modify(|s| s.status = JobStatus::Running);
        let started = Instant::now();
        let mut timings = BTreeMap::new();
        timings.insert(
            "queued".to_string(),
            started.duration_since(entry.submitted_at).as_secs_f64(),
        );

        let backend_id = job.model.backend.clone();
        let Some(backend) = self.deps.backends.get(&backend_id) else {
            let reason = self.deps.backends.availability_of(&job.model).await;
            return entry.fail(&SfxError::BackendUnavailable {
                backend: backend_id,
                reason: reason.reason().unwrap_or("not available").to_string(),
            });
        };
        let runs = backend.runs();

        // Local lane only: the cross-process GPU lock, one local model resident, the memory preflight. The cloud lane
        // holds no local memory, so it needs none of them.
        let mut lock = None;
        if runs == Runs::Local {
            let t0 = Instant::now();
            let holder = Holder::this_process(Some(id));
            match self
                .deps
                .gpu_lock
                .acquire(&holder, &entry.cancel, |current| {
                    entry.set_stage(Stage::WaitingGpuLock {
                        holder: current.map(|h| h.describe()),
                    });
                })
                .await
            {
                Ok(guard) => lock = Some(guard),
                Err(err) => return entry.fail(&err),
            }
            timings.insert("gpu_lock_wait".into(), t0.elapsed().as_secs_f64());

            for other in self.deps.backends.all() {
                if other.id() != backend.id()
                    && other.runs() == Runs::Local
                    && other.loaded_model().is_some()
                {
                    other.unload().await;
                }
            }
            if backend.loaded_model().as_deref() != Some(job.model.id.as_str()) {
                let deps = &self.deps;
                if let Err(err) = preflight(
                    &job.model,
                    deps.memory_headroom_mb,
                    deps.memory.as_ref(),
                    || advice(&census(&deps.processes.processes(), &deps.heavy_processes)),
                ) {
                    return entry.fail(&err);
                }
            }
        }

        // From here on every exit path calls `abort` and removes the scratch folder.
        if let Err(err) = job.finisher.prepare().await {
            return entry.fail(&err);
        }
        let scratch = self.deps.scratch_dir.join(id);
        let outcome = self
            .generate_and_finish(
                id, entry, &job, &scratch, backend, runs, &mut lock, timings, started,
            )
            .await;
        drop(lock);
        let _ = std::fs::remove_dir_all(&scratch);
        match outcome {
            Ok(result) => entry.finish(JobStatus::Done, Some(result), None),
            Err(err) => {
                job.finisher.abort().await;
                entry.fail(&err);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_and_finish(
        &self,
        id: &str,
        entry: &JobEntry,
        job: &JobRequest,
        scratch: &std::path::Path,
        backend: Arc<dyn crate::backend::SfxBackend>,
        runs: Runs,
        lock: &mut Option<crate::gpu_lock::GpuLockGuard>,
        mut timings: BTreeMap<String, f64>,
        started: Instant,
    ) -> Result<Value, SfxError> {
        std::fs::create_dir_all(scratch).map_err(|e| SfxError::io(scratch, e))?;
        let mut request = job.request.clone();
        request.out_dir = scratch.to_path_buf();

        let tx = entry.tx.clone();
        let progress = ProgressSink::new(move |stage: Stage| {
            tx.send_modify(|s| {
                s.progress_text = Some(stage.to_string());
                s.progress = Some(stage);
            });
        });
        let t0 = Instant::now();
        let outcome = backend
            .generate(&job.model, &request, progress.clone(), entry.cancel.clone())
            .await;
        // The GPU is free again; post-processing is CPU work and must not hold up sister servers.
        lock.take();
        timings.insert("generate".into(), t0.elapsed().as_secs_f64());
        let clips = match outcome {
            _ if entry.cancel.is_cancelled() => return Err(SfxError::Cancelled),
            Err(BackendError::Cancelled) => return Err(SfxError::Cancelled),
            Err(e) => return Err(e.into_error(backend.id())),
            Ok(clips) => clips,
        };

        let ctx = FinishContext {
            job_id: id,
            model: &job.model,
            request: &request,
            backend_id: backend.id(),
            runs,
            provider: backend.provider(),
            timings: {
                timings.insert("total_so_far".into(), started.elapsed().as_secs_f64());
                timings
            },
            progress: &progress,
            cancel: &entry.cancel,
        };
        let result = job.finisher.finish(clips, &ctx).await?;
        if entry.cancel.is_cancelled() {
            return Err(SfxError::Cancelled);
        }
        Ok(result)
    }
}

async fn worker(inner: std::sync::Weak<Inner>, lane: Lane) {
    let mut idle_armed = false;
    loop {
        let Some(this) = inner.upgrade() else { return };
        if let Some((id, entry)) = this.next_job(lane) {
            this.run(&id, &entry).await;
            *this.lane(lane).running.lock().unwrap() = None;
            // Only local work leaves models resident, so only the local lane arms the idle unload.
            idle_armed = lane == Lane::Local && !this.deps.idle_unload.is_zero();
            continue;
        }
        let idle = this.deps.idle_unload;
        // `notify_one` stores a permit, so a job submitted between the check above and this wait isn't missed.
        let wake = async {
            this.lane(lane).wake.notified().await;
        };
        if idle_armed {
            tokio::select! {
                _ = wake => {}
                _ = tokio::time::sleep(idle) => {
                    this.unload_all().await;
                    idle_armed = false;
                }
            }
        } else {
            wake.await;
        }
    }
}

/// If the call that waits for a job is dropped (a cancelled request or Task), stop the job too.
pub struct CancelOnDrop {
    jobs: JobManager,
    job_id: Option<String>,
}

impl CancelOnDrop {
    pub fn new(jobs: JobManager, job_id: impl Into<String>) -> Self {
        Self {
            jobs,
            job_id: Some(job_id.into()),
        }
    }

    /// The call finished normally: don't cancel.
    pub fn disarm(mut self) {
        self.job_id = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(id) = self.job_id.take() {
            tracing::info!(job_id = %id, "request dropped before completion; cancelling job");
            let _ = self.jobs.cancel(&id);
        }
    }
}

fn to_job_error(err: &SfxError) -> JobError {
    let (stage, kind) = match err {
        SfxError::Backend { failure, .. } => {
            (Some(failure.stage.clone()), Some(failure.kind.clone()))
        }
        _ => (None, None),
    };
    JobError {
        code: err.code().to_string(),
        message: err.scrubbed_message(),
        caller_error: err.is_caller_error(),
        fix: crate::secrets::scrub(&err.fix()),
        stage,
        kind,
    }
}

fn new_job_id() -> String {
    format!("job_{:016x}", rand::random::<u64>())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use super::*;
    use crate::backend::SfxBackend;
    use crate::backend::mock::{MOCK_MODELS, MockBackend};
    use crate::heavy::{FixedProcesses, ProcessInfo};
    use crate::registry::ModelRegistry;

    const STEP: Duration = Duration::from_millis(100);

    /// Counts the calls it gets; optionally fails `finish`, or checks that the GPU lock is free while it runs.
    #[derive(Default)]
    struct TestFinisher {
        prepared: AtomicUsize,
        finished: AtomicUsize,
        aborted: AtomicUsize,
        fail_finish: AtomicBool,
        fail_prepare: AtomicBool,
        probe_lock: Mutex<Option<GpuLock>>,
        lock_was_free: Mutex<Option<bool>>,
        scratch_seen: Mutex<Vec<PathBuf>>,
    }

    #[async_trait::async_trait]
    impl JobFinisher for TestFinisher {
        async fn prepare(&self) -> Result<(), SfxError> {
            self.prepared.fetch_add(1, Ordering::SeqCst);
            if self.fail_prepare.load(Ordering::SeqCst) {
                return Err(SfxError::OutputExists {
                    path: "/x/a_01.wav".into(),
                });
            }
            Ok(())
        }

        async fn finish(
            &self,
            clips: Vec<RawClip>,
            ctx: &FinishContext<'_>,
        ) -> Result<Value, SfxError> {
            self.finished.fetch_add(1, Ordering::SeqCst);
            ctx.progress.report(Stage::PostProcessing);
            self.scratch_seen
                .lock()
                .unwrap()
                .push(ctx.request.out_dir.clone());
            if let Some(lock) = self.probe_lock.lock().unwrap().as_ref() {
                let free = lock
                    .try_acquire(&Holder::this_process(None))
                    .unwrap()
                    .is_some();
                *self.lock_was_free.lock().unwrap() = Some(free);
            }
            if self.fail_finish.load(Ordering::SeqCst) {
                return Err(SfxError::io(std::path::Path::new("/x"), "disk full"));
            }
            SummaryFinisher.finish(clips, ctx).await
        }

        async fn abort(&self) {
            self.aborted.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct Rig {
        dir: tempfile::TempDir,
        jobs: JobManager,
        local: Arc<MockBackend>,
        cloud: Arc<MockBackend>,
        memory: Arc<AtomicU64>,
    }

    struct SharedMemory(Arc<AtomicU64>);

    impl MemoryProbe for SharedMemory {
        fn available_mb(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn model(id: &str) -> ModelSpec {
        ModelRegistry::from_toml(MOCK_MODELS)
            .unwrap()
            .get(id)
            .unwrap()
            .clone()
    }

    fn rig_with(queue_depth: usize, memory_mb: u64, idle: Duration) -> Rig {
        let dir = tempfile::tempdir().unwrap();
        let local = Arc::new(MockBackend::local(STEP));
        let cloud = Arc::new(MockBackend::cloud(STEP));
        let mut backends = BackendRegistry::default();
        backends.insert(local.clone());
        backends.insert(cloud.clone());
        let memory = Arc::new(AtomicU64::new(memory_mb));
        let jobs = JobManager::new(JobDeps {
            backends,
            gpu_lock: GpuLock::new(dir.path().join("lock/local-gpu.lock")),
            memory: Arc::new(SharedMemory(memory.clone())),
            processes: Arc::new(FixedProcesses(vec![ProcessInfo {
                name: "UnrealEditor".into(),
                cmdline: String::new(),
                resident_mb: 6100,
            }])),
            heavy_processes: vec!["UnrealEditor".into()],
            memory_headroom_mb: 1000,
            queue_depth,
            job_ttl: Duration::from_secs(10),
            idle_unload: idle,
            scratch_dir: dir.path().join("scratch"),
        });
        Rig {
            dir,
            jobs,
            local,
            cloud,
            memory,
        }
    }

    fn rig() -> Rig {
        rig_with(4, 64_000, Duration::from_secs(5))
    }

    fn lock_of(rig: &Rig) -> GpuLock {
        GpuLock::new(rig.dir.path().join("lock/local-gpu.lock"))
    }

    fn job_for(model_id: &str, variations: u32) -> (JobRequest, Arc<TestFinisher>) {
        let finisher = Arc::new(TestFinisher::default());
        let job = JobRequest {
            model: model(model_id),
            request: BackendRequest {
                prompt: "a horn".into(),
                duration_s: Some(0.5),
                variations,
                seed: Some(7),
                sample_rate: 44_100,
                ..Default::default()
            },
            finisher: finisher.clone(),
        };
        (job, finisher)
    }

    fn local_job(variations: u32) -> (JobRequest, Arc<TestFinisher>) {
        job_for("mock-local", variations)
    }

    fn cloud_job(variations: u32) -> (JobRequest, Arc<TestFinisher>) {
        job_for("mock-cloud", variations)
    }

    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    fn scratch_is_empty(rig: &Rig) -> bool {
        std::fs::read_dir(rig.dir.path().join("scratch"))
            .map(|mut d| d.next().is_none())
            .unwrap_or(true)
    }

    #[tokio::test(start_paused = true)]
    async fn runs_one_at_a_time_in_order_and_stores_the_finishers_value() {
        let rig = rig();
        let (ja, fa) = local_job(2);
        let (jb, _) = local_job(1);
        let a = rig.jobs.submit(ja).unwrap();
        let b = rig.jobs.submit(jb).unwrap();
        assert_eq!(a.lane, Lane::Local);
        assert_eq!(a.progress, Some(Stage::Queued { position: 1 }));
        settle().await;
        assert_eq!(rig.jobs.get(&a.job_id).unwrap().status, JobStatus::Running);
        let b_now = rig.jobs.get(&b.job_id).unwrap();
        assert_eq!(b_now.status, JobStatus::Queued);
        assert_eq!(b_now.progress, Some(Stage::Queued { position: 1 }));

        let a_done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(a_done.status, JobStatus::Done);
        let b_done = rig.jobs.wait(&b.job_id).await.unwrap();
        assert_eq!(b_done.status, JobStatus::Done);
        let result = a_done.result.unwrap();
        assert_eq!(result["model"], "mock-local");
        assert_eq!(result["runs"], "local");
        assert_eq!(result["clips"].as_array().unwrap().len(), 2);
        assert_eq!(result["clips"][1]["seed"], 8);
        assert!(result["timings"]["generate"].is_number());
        assert!(result["timings"]["gpu_lock_wait"].is_number());
        assert_eq!(
            (
                fa.prepared.load(Ordering::SeqCst),
                fa.finished.load(Ordering::SeqCst),
                fa.aborted.load(Ordering::SeqCst)
            ),
            (1, 1, 0)
        );
        assert!(scratch_is_empty(&rig), "scratch folders are removed");
        // The terminal snapshot carries no stale progress.
        assert_eq!(rig.jobs.get(&a.job_id).unwrap().progress, None);
    }

    #[tokio::test(start_paused = true)]
    async fn the_backend_gets_a_scratch_folder_per_job() {
        let rig = rig();
        let (job, finisher) = local_job(1);
        let snap = rig.jobs.submit(job).unwrap();
        rig.jobs.wait(&snap.job_id).await.unwrap();
        let seen = finisher.scratch_seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], rig.dir.path().join("scratch").join(&snap.job_id));
        assert!(!seen[0].exists(), "removed after the job");
    }

    #[tokio::test(start_paused = true)]
    async fn queue_full_is_per_lane() {
        let rig = rig_with(1, 64_000, Duration::from_secs(5));
        rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.submit(cloud_job(1).0).unwrap();
        settle().await; // both running, both queues empty
        rig.jobs.submit(local_job(1).0).unwrap();
        let c2 = rig.jobs.submit(cloud_job(1).0).unwrap();
        assert_eq!(c2.progress, Some(Stage::Queued { position: 1 }));
        assert_eq!(
            rig.jobs.submit(local_job(1).0).unwrap_err().code(),
            "queue_full"
        );
        assert_eq!(
            rig.jobs.submit(cloud_job(1).0).unwrap_err().code(),
            "queue_full"
        );
        assert_eq!(rig.jobs.busy(), 4);
        assert_eq!(
            (rig.jobs.busy_in(Lane::Local), rig.jobs.busy_in(Lane::Cloud)),
            (2, 2)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_queued_and_running_leave_nothing_behind() {
        let rig = rig();
        let (ja, fa) = local_job(4);
        let (jb, fb) = local_job(4);
        let a = rig.jobs.submit(ja).unwrap();
        let b = rig.jobs.submit(jb).unwrap();
        settle().await;

        let b_snap = rig.jobs.cancel(&b.job_id).unwrap();
        assert_eq!(b_snap.status, JobStatus::Cancelled);
        assert_eq!(
            fb.prepared.load(Ordering::SeqCst),
            0,
            "a queued job never prepares"
        );

        tokio::time::sleep(STEP * 3).await;
        rig.jobs.cancel(&a.job_id).unwrap();
        let a_done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(a_done.status, JobStatus::Cancelled);
        assert!(a_done.error.is_none());
        assert_eq!(
            (
                fa.prepared.load(Ordering::SeqCst),
                fa.finished.load(Ordering::SeqCst),
                fa.aborted.load(Ordering::SeqCst)
            ),
            (1, 0, 1),
            "abort cleans the reservation"
        );
        assert_eq!(rig.local.generations.load(Ordering::SeqCst), 0);
        assert!(scratch_is_empty(&rig));
        // The lock was released: another job can take it.
        assert!(
            lock_of(&rig)
                .try_acquire(&Holder::this_process(None))
                .unwrap()
                .is_some()
        );
        // Cancelling again is harmless.
        assert_eq!(
            rig.jobs.cancel(&a.job_id).unwrap().status,
            JobStatus::Cancelled
        );
    }

    #[tokio::test(start_paused = true)]
    async fn backend_failure_is_reported_with_stage_and_kind() {
        let rig = rig();
        rig.local.fail_next("generating", "worker_crashed");
        let (job, finisher) = local_job(1);
        let a = rig.jobs.submit(job).unwrap();
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        let err = done.error.unwrap();
        assert_eq!(err.code, "worker_crashed");
        assert_eq!(err.stage.as_deref(), Some("generating"));
        assert_eq!(err.kind.as_deref(), Some("worker_crashed"));
        assert!(
            err.message.contains("injected worker_crashed"),
            "{}",
            err.message
        );
        assert_eq!(finisher.aborted.load(Ordering::SeqCst), 1);
        assert_eq!(finisher.finished.load(Ordering::SeqCst), 0);
        assert!(scratch_is_empty(&rig));
    }

    #[tokio::test(start_paused = true)]
    async fn finisher_failures_fail_the_job_and_abort() {
        let rig = rig();
        let (job, finisher) = local_job(1);
        finisher.fail_finish.store(true, Ordering::SeqCst);
        let a = rig.jobs.submit(job).unwrap();
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        assert_eq!(done.error.unwrap().code, "io_error");
        assert_eq!(finisher.aborted.load(Ordering::SeqCst), 1);

        // A failing prepare (names taken) fails before the backend runs, with nothing to abort.
        let (job, finisher) = local_job(1);
        finisher.fail_prepare.store(true, Ordering::SeqCst);
        let a = rig.jobs.submit(job).unwrap();
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.error.unwrap().code, "output_exists");
        assert_eq!(finisher.aborted.load(Ordering::SeqCst), 0);
        assert_eq!(
            rig.local.generations.load(Ordering::SeqCst),
            1,
            "only the first job generated"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_gpu_lock_is_released_before_the_finisher_runs() {
        let rig = rig();
        let (job, finisher) = local_job(1);
        *finisher.probe_lock.lock().unwrap() = Some(lock_of(&rig));
        let a = rig.jobs.submit(job).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(*finisher.lock_was_free.lock().unwrap(), Some(true));
    }

    #[tokio::test(start_paused = true)]
    async fn finished_jobs_expire() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        tokio::time::sleep(Duration::from_secs(9)).await;
        assert!(rig.jobs.get(&a.job_id).is_ok());
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(rig.jobs.get(&a.job_id).unwrap_err().code(), "unknown_job");
    }

    #[tokio::test(start_paused = true)]
    async fn idle_unload_fires_once_and_only_for_local_work() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        assert!(rig.local.loaded_model().is_some());
        tokio::time::sleep(Duration::from_secs(4)).await;
        assert_eq!(rig.local.unloads.load(Ordering::SeqCst), 0);
        tokio::time::sleep(Duration::from_secs(2)).await;
        settle().await;
        assert_eq!(rig.local.unloads.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_secs(60)).await;
        assert_eq!(rig.local.unloads.load(Ordering::SeqCst), 1);

        // Cloud work arms nothing.
        let c = rig.jobs.submit(cloud_job(1).0).unwrap();
        rig.jobs.wait(&c.job_id).await.unwrap();
        tokio::time::sleep(Duration::from_secs(60)).await;
        assert_eq!(rig.cloud.unloads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn zero_idle_means_never() {
        let rig = rig_with(4, 64_000, Duration::ZERO);
        let a = rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        tokio::time::sleep(Duration::from_secs(3600)).await;
        assert_eq!(rig.local.unloads.load(Ordering::SeqCst), 0);
        assert!(rig.local.loaded_model().is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn unload_all_is_refused_while_a_local_job_is_busy() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(4).0).unwrap();
        settle().await;
        let err = rig.jobs.unload_all().await.unwrap_err();
        assert_eq!(err.code(), "invalid_params");
        rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(
            rig.jobs.unload_all().await.unwrap(),
            vec!["mock-local".to_string()]
        );
        assert_eq!(rig.local.unloads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn unload_is_allowed_while_only_a_cloud_job_runs() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        let c = rig.jobs.submit(cloud_job(4).0).unwrap();
        settle().await;
        assert_eq!(rig.jobs.get(&c.job_id).unwrap().status, JobStatus::Running);
        assert_eq!(
            rig.jobs.unload_all().await.unwrap(),
            vec!["mock-local".to_string()]
        );
        assert_eq!(
            rig.jobs.wait(&c.job_id).await.unwrap().status,
            JobStatus::Done
        );
    }

    #[tokio::test(start_paused = true)]
    async fn progress_is_streamed_with_generating_n_of_n() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(3).0).unwrap();
        let mut rx = rig.jobs.subscribe(&a.job_id).unwrap();
        let mut seen = Vec::new();
        while rx.changed().await.is_ok() {
            let snap = rx.borrow_and_update().clone();
            if let Some(text) = snap.progress_text.clone() {
                seen.push(text);
            }
            if snap.status.is_terminal() {
                break;
            }
        }
        // watch keeps only the latest value, so observers see a subset of the stages.
        assert!(
            seen.iter().any(|s| s.starts_with("generating ")),
            "{seen:?}"
        );
        assert!(seen.iter().any(|s| s.ends_with("of 3")), "{seen:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn wait_with_progress_returns_the_terminal_snapshot() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(2).0).unwrap();
        let done = rig.jobs.wait_with_progress(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Done);
        let err = rig.jobs.wait_with_progress("job_nope").await.unwrap_err();
        assert_eq!(err.code(), "unknown_job");
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_on_drop_cancels_unless_disarmed() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(4).0).unwrap();
        settle().await;
        drop(CancelOnDrop::new(rig.jobs.clone(), a.job_id.clone()));
        assert_eq!(
            rig.jobs.wait(&a.job_id).await.unwrap().status,
            JobStatus::Cancelled
        );

        let b = rig.jobs.submit(local_job(1).0).unwrap();
        let guard = CancelOnDrop::new(rig.jobs.clone(), b.job_id.clone());
        guard.disarm();
        assert_eq!(
            rig.jobs.wait(&b.job_id).await.unwrap().status,
            JobStatus::Done
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_all_stops_both_lanes() {
        let rig = rig();
        let ids: Vec<String> = [local_job(4).0, local_job(4).0, cloud_job(4).0]
            .into_iter()
            .map(|j| rig.jobs.submit(j).unwrap().job_id)
            .collect();
        settle().await;
        rig.jobs.cancel_all();
        for id in &ids {
            assert_eq!(
                rig.jobs.wait(id).await.unwrap().status,
                JobStatus::Cancelled,
                "{id}"
            );
        }
        assert_eq!(rig.jobs.busy(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_cloud_job_runs_while_a_local_job_is_busy() {
        let rig = rig();
        let local = rig.jobs.submit(local_job(8).0).unwrap();
        let (cjob, _) = cloud_job(2);
        let cloud = rig.jobs.submit(cjob).unwrap();
        assert_eq!(cloud.lane, Lane::Cloud);
        settle().await;
        // Both lanes start at once: the cloud job isn't queued behind the local one.
        assert_eq!(
            rig.jobs.get(&cloud.job_id).unwrap().status,
            JobStatus::Running
        );
        assert_eq!(
            rig.jobs.get(&local.job_id).unwrap().status,
            JobStatus::Running
        );
        let done = rig.jobs.wait(&cloud.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Done);
        assert_eq!(
            rig.jobs.get(&local.job_id).unwrap().status,
            JobStatus::Running
        );
        assert_eq!(
            (rig.jobs.busy_in(Lane::Local), rig.jobs.busy_in(Lane::Cloud)),
            (1, 0)
        );

        let result = done.result.unwrap();
        assert_eq!(result["runs"], "cloud");
        assert!(result["clips"][0]["seed"].is_null());
        assert_eq!(result["clips"][0]["credits"], 5);
        assert!(
            result["timings"].get("gpu_lock_wait").is_none(),
            "no lock in the cloud lane"
        );
        assert_eq!(
            rig.jobs.wait(&local.job_id).await.unwrap().status,
            JobStatus::Done
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_cloud_lane_ignores_the_gpu_lock_and_memory() {
        let rig = rig_with(4, 10, Duration::from_secs(5)); // 10 MB free: any local model is refused
        let holder = Holder {
            server: "mcp-meshgen".into(),
            pid: 4242,
            job: None,
            since_unix: 0,
        };
        let _held = lock_of(&rig).try_acquire(&holder).unwrap().unwrap();
        let c = rig.jobs.submit(cloud_job(1).0).unwrap();
        let done = rig.jobs.wait(&c.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Done, "{:?}", done.error);
    }

    #[tokio::test(start_paused = true)]
    async fn local_jobs_wait_for_the_gpu_lock_and_say_who_holds_it() {
        let rig = rig();
        let holder = Holder {
            server: "mcp-meshgen".into(),
            pid: 4242,
            job: Some("m1".into()),
            since_unix: 0,
        };
        let guard = lock_of(&rig).try_acquire(&holder).unwrap().unwrap();
        let (job, finisher) = local_job(1);
        let a = rig.jobs.submit(job).unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        let snap = rig.jobs.get(&a.job_id).unwrap();
        assert_eq!(snap.status, JobStatus::Running);
        assert_eq!(
            snap.progress_text.as_deref(),
            Some("waiting for another local model (mcp-meshgen (pid 4242))")
        );
        assert_eq!(
            snap.progress,
            Some(Stage::WaitingGpuLock {
                holder: Some("mcp-meshgen (pid 4242)".into())
            })
        );
        assert_eq!(
            finisher.prepared.load(Ordering::SeqCst),
            0,
            "nothing reserved while waiting"
        );
        drop(guard);
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Done);
        assert!(
            done.result.unwrap()["timings"]["gpu_lock_wait"]
                .as_f64()
                .unwrap()
                >= 1.0
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_while_waiting_for_the_lock() {
        let rig = rig();
        let _guard = lock_of(&rig)
            .try_acquire(&Holder::this_process(None))
            .unwrap()
            .unwrap();
        let (job, finisher) = local_job(1);
        let a = rig.jobs.submit(job).unwrap();
        tokio::time::sleep(Duration::from_secs(2)).await;
        rig.jobs.cancel(&a.job_id).unwrap();
        assert_eq!(
            rig.jobs.wait(&a.job_id).await.unwrap().status,
            JobStatus::Cancelled
        );
        assert_eq!(finisher.aborted.load(Ordering::SeqCst), 0);
        assert_eq!(rig.local.generations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn the_preflight_refuses_a_local_job_and_names_what_to_close() {
        // mock-local needs 100 + 1000 headroom.
        let rig = rig_with(4, 500, Duration::from_secs(5));
        let (job, finisher) = local_job(1);
        let a = rig.jobs.submit(job).unwrap();
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        let err = done.error.unwrap();
        assert_eq!(err.code, "out_of_memory");
        assert!(err.message.contains("1100 MB"), "{}", err.message);
        assert!(
            err.message.contains("Close UnrealEditor (6100 MB)"),
            "{}",
            err.message
        );
        assert_eq!(finisher.prepared.load(Ordering::SeqCst), 0);
        assert_eq!(rig.local.generations.load(Ordering::SeqCst), 0);
        // The lock was released on the refusal.
        assert!(
            lock_of(&rig)
                .try_acquire(&Holder::this_process(None))
                .unwrap()
                .is_some()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_preflight_is_skipped_when_the_model_is_already_resident() {
        let rig = rig();
        let a = rig.jobs.submit(local_job(1).0).unwrap();
        rig.jobs.wait(&a.job_id).await.unwrap();
        assert!(rig.local.loaded_model().is_some());
        // Memory has since gone: a resident worker needs no new memory.
        rig.memory.store(0, Ordering::SeqCst);
        let b = rig.jobs.submit(local_job(1).0).unwrap();
        assert_eq!(
            rig.jobs.wait(&b.job_id).await.unwrap().status,
            JobStatus::Done
        );
        // After an unload the check applies again.
        rig.jobs.unload_all().await.unwrap();
        let c = rig.jobs.submit(local_job(1).0).unwrap();
        let done = rig.jobs.wait(&c.job_id).await.unwrap();
        assert_eq!(done.error.unwrap().code, "out_of_memory");
    }

    #[tokio::test(start_paused = true)]
    async fn a_local_job_unloads_another_resident_local_backend() {
        let dir = tempfile::tempdir().unwrap();
        let first = Arc::new(MockBackend::new(
            "mock",
            Runs::Local,
            Duration::from_millis(1),
        ));
        let second = Arc::new(MockBackend::new(
            "mock-b",
            Runs::Local,
            Duration::from_millis(1),
        ));
        let mut backends = BackendRegistry::default();
        backends.insert(first.clone());
        backends.insert(second.clone());
        let jobs = JobManager::new(JobDeps {
            backends,
            gpu_lock: GpuLock::new(dir.path().join("l/local-gpu.lock")),
            memory: Arc::new(crate::memory::FixedMemory(64_000)),
            processes: Arc::new(FixedProcesses(vec![])),
            heavy_processes: vec![],
            memory_headroom_mb: 0,
            queue_depth: 4,
            job_ttl: Duration::from_secs(10),
            idle_unload: Duration::ZERO,
            scratch_dir: dir.path().join("scratch"),
        });
        let a = jobs.submit(local_job(1).0).unwrap();
        jobs.wait(&a.job_id).await.unwrap();
        assert!(first.loaded_model().is_some());
        let (mut job, _) = local_job(1);
        job.model.backend = "mock-b".into();
        job.model.id = "mock-local-b".into();
        let b = jobs.submit(job).unwrap();
        jobs.wait(&b.job_id).await.unwrap();
        assert!(
            first.loaded_model().is_none(),
            "only one local model resident at a time"
        );
        assert_eq!(second.loaded_model().as_deref(), Some("mock-local-b"));
    }

    #[tokio::test(start_paused = true)]
    async fn an_unregistered_backend_fails_the_job_with_a_reason() {
        let rig = rig();
        let (mut job, finisher) = local_job(1);
        job.model.backend = "sa3_mlx".into();
        let a = rig.jobs.submit(job).unwrap();
        assert_eq!(a.lane, Lane::Local);
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        let err = done.error.unwrap();
        assert_eq!(err.code, "backend_unavailable");
        assert!(err.message.contains("sa3_mlx"), "{}", err.message);
        assert_eq!(finisher.prepared.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn unknown_jobs_are_caller_errors() {
        let rig = rig();
        let err = rig.jobs.get("job_nope").unwrap_err();
        assert_eq!(err.code(), "unknown_job");
        assert!(err.is_caller_error());
        assert_eq!(
            rig.jobs.cancel("job_nope").unwrap_err().code(),
            "unknown_job"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_variations_all_run_in_one_job() {
        // The cloud lane runs one job at a time; a job's own variations are the backend's business.
        let rig = rig();
        let (job, _) = cloud_job(4);
        let a = rig.jobs.submit(job).unwrap();
        let done = rig.jobs.wait(&a.job_id).await.unwrap();
        assert_eq!(done.result.unwrap()["clips"].as_array().unwrap().len(), 4);
        assert_eq!(rig.cloud.max_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn snapshots_serialize_with_lane_and_stage() {
        let snap = JobSnapshot {
            job_id: "j".into(),
            status: JobStatus::Running,
            lane: Lane::Cloud,
            progress: Some(Stage::Generating { n: 1, of: 2 }),
            progress_text: Some("generating 1 of 2".into()),
            result: None,
            error: None,
        };
        let json = serde_json::to_value(&snap).unwrap();
        assert_eq!(json["lane"], "cloud");
        assert_eq!(json["status"], "running");
        assert_eq!(json["progress"]["stage"], "generating");
        assert!(json.get("result").is_none());
    }

    #[allow(dead_code)]
    fn assert_backend_is_object_safe(_: &dyn SfxBackend) {}
}
