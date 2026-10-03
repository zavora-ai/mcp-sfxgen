//! Stable Audio 3 small-sfx on Apple Silicon through MLX (design §4.2; R2, R8, R11).
//!
//! Runs `worker/sa3_mlx_worker.py` (embedded here, written to the cache dir) with the stable-audio-3 venv's python, in its
//! own process group, and talks JSON lines to it (`docs/worker-protocol.md`). The worker stays loaded between jobs
//! (about 2.4 GB resident) and is stopped by `unload_models`, after `idle_unload_s` idle seconds, or when a job fails.
//! Result files are raw interleaved little-endian f32; they are read into memory and deleted straight away, so the
//! scratch folder is empty again when `generate` returns, whatever the outcome.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::process::{kill_tree, own_process_group};
use super::{
    Availability, BackendError, BackendRequest, ProgressSink, RawClip, Runs, SfxBackend, Stage,
};
use crate::audio::Clip;
use crate::config::{Sa3MlxConfig, default_hf_cache};
use crate::registry::{BackendOptions, ModelSpec};

pub const WORKER_SOURCE: &str = include_str!("../../worker/sa3_mlx_worker.py");
pub const PROTOCOL: u64 = 1;
/// Length of a clip when the caller gave none (design §5: the local default).
pub const DEFAULT_DURATION_S: f64 = crate::request::LOCAL_DEFAULT_DURATION_S;
const LOG_TAIL_LINES: usize = 40;
/// The worker says hello right after the Python imports and the weight check, before any weight is loaded.
const DEFAULT_HELLO_TIMEOUT: Duration = Duration::from_secs(60);
const EXIT_TIMEOUT: Duration = Duration::from_secs(15);
const OOM_HINT: &str = "close memory-heavy apps (Unreal Editor, Blender, other local models) and retry; fewer variations or a \
shorter sound also needs less memory";

/// How the worker is started: the embedded Python worker, or a stand-in program (tests: `fake-worker`).
#[derive(Debug, Clone)]
pub enum Launcher {
    Python,
    Program {
        path: PathBuf,
        env: Vec<(String, String)>,
    },
}

pub struct Sa3MlxBackend {
    config: Sa3MlxConfig,
    hf_cache: PathBuf,
    cache_dir: PathBuf,
    idle_unload_s: u64,
    launcher: Launcher,
    hello_timeout: Duration,
    slot: Arc<tokio::sync::Mutex<Option<Worker>>>,
    loaded: Arc<Mutex<Option<String>>>,
    idle_generation: Arc<AtomicU64>,
}

struct Worker {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Lines<BufReader<ChildStdout>>,
    log: Arc<Mutex<VecDeque<String>>>,
    log_task: Option<JoinHandle<()>>,
    /// What the worker said about itself in `hello` (mlx, python, revision), for the sidecar.
    hello: serde_json::Value,
}

#[derive(Debug, Default, Deserialize)]
struct Event {
    event: String,
    #[serde(default)]
    id: String,
    #[serde(default)]
    protocol: u64,
    #[serde(default)]
    stage: String,
    #[serde(default)]
    step: u32,
    #[serde(default)]
    total: u32,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    message: String,
    #[serde(default)]
    hint: String,
    #[serde(default)]
    sample_rate: u32,
    #[serde(default)]
    channels: u16,
    #[serde(default)]
    frames: u64,
    #[serde(default)]
    files: Vec<PathBuf>,
    #[serde(default)]
    seeds: Vec<u64>,
    #[serde(default)]
    timings: BTreeMap<String, f64>,
    #[serde(default)]
    peak_mlx_mb: Option<f64>,
    #[serde(default)]
    active_mlx_mb: Option<f64>,
    #[serde(default)]
    peak_rss_mb: Option<f64>,
    #[serde(default)]
    mlx: Option<String>,
    #[serde(default)]
    python: Option<String>,
    #[serde(default)]
    revision: Option<String>,
}

impl Sa3MlxBackend {
    pub fn new(
        config: Sa3MlxConfig,
        hf_cache: PathBuf,
        cache_dir: PathBuf,
        idle_unload_s: u64,
    ) -> Self {
        Self::with_launcher(config, hf_cache, cache_dir, idle_unload_s, Launcher::Python)
    }

    pub fn with_launcher(
        config: Sa3MlxConfig,
        hf_cache: PathBuf,
        cache_dir: PathBuf,
        idle_unload_s: u64,
        launcher: Launcher,
    ) -> Self {
        Self {
            config,
            hf_cache,
            cache_dir,
            idle_unload_s,
            launcher,
            hello_timeout: DEFAULT_HELLO_TIMEOUT,
            slot: Arc::new(tokio::sync::Mutex::new(None)),
            loaded: Arc::new(Mutex::new(None)),
            idle_generation: Arc::new(AtomicU64::new(0)),
        }
    }

    /// How long to wait for the worker's `hello` (tests shorten it).
    pub fn with_hello_timeout(mut self, timeout: Duration) -> Self {
        self.hello_timeout = timeout;
        self
    }

    /// Availability rules (§4.2), with the platform passed in so every branch is testable. File checks only: nothing is
    /// loaded.
    pub fn check(&self, model: &ModelSpec, apple_silicon: bool) -> Availability {
        let unavailable = |reason: String| Availability::Unavailable { reason };
        if let Launcher::Program { path, .. } = &self.launcher {
            return if path.is_file() {
                Availability::Ready
            } else {
                unavailable(format!("worker program {} not found", path.display()))
            };
        }
        if !apple_silicon {
            return unavailable(
                "Stable Audio 3 through MLX needs Apple Silicon (macOS on arm64); use a cloud model on this machine".into(),
            );
        }
        let python = self.config.python_path();
        if !python.is_file() {
            return unavailable(format!(
                "no venv python at {}; set up the stable-audio-3 MLX checkout (its install.sh creates .venv) or set \
                 [sa3_mlx] dir in config.toml to it",
                python.display()
            ));
        }
        let BackendOptions::Sa3Mlx {
            dit, decoder, t5, ..
        } = &model.options
        else {
            return unavailable(format!(
                "model `{}` has no sa3_mlx options in models.toml",
                model.id
            ));
        };
        let folder = self.config.dir.join("models").join("mlx");
        // `metadata` follows symlinks, so a link into the Hugging Face cache that dangles counts as missing.
        let missing: Vec<&str> = [dit, decoder, t5]
            .into_iter()
            .map(String::as_str)
            .filter(|name| std::fs::metadata(folder.join(name)).is_err())
            .collect();
        if !missing.is_empty() {
            let download = model.missing_files(&self.hf_cache, &default_hf_cache());
            let how = if download.is_empty() {
                format!(
                    "they are in the Hugging Face cache but not linked: link them into the folder, or rerun the \
                     stable-audio-3 installer in {}",
                    self.config.dir.display()
                )
            } else {
                format!(
                    "they aren't downloaded yet (needs your OK): {}",
                    download
                        .iter()
                        .map(|m| m.command.clone())
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            };
            return unavailable(format!(
                "missing weight file(s) {} under {}; {how}",
                missing.join(", "),
                folder.display()
            ));
        }
        Availability::Ready
    }

    /// Write the embedded worker to `<cache>/worker/sa3_mlx_worker-<hash>.py` (once per version).
    pub fn worker_script(&self) -> Result<PathBuf, BackendError> {
        let hash = format!("{:x}", Sha256::digest(WORKER_SOURCE.as_bytes()));
        let dir = self.cache_dir.join("worker");
        let path = dir.join(format!("sa3_mlx_worker-{}.py", &hash[..12]));
        if path.is_file() {
            return Ok(path);
        }
        let io = |e: std::io::Error, p: &Path| {
            BackendError::failed(
                "starting",
                "cache_write",
                format!("{}: {e}", p.display()),
                "check the cache dir is writable",
            )
        };
        std::fs::create_dir_all(&dir).map_err(|e| io(e, &dir))?;
        // Unique per writer: two threads (or servers) may write the same script at once.
        let tmp = dir.join(format!(
            ".sa3_mlx_worker-{}.{:016x}.tmp",
            &hash[..12],
            rand::random::<u64>()
        ));
        std::fs::write(&tmp, WORKER_SOURCE).map_err(|e| io(e, &tmp))?;
        std::fs::rename(&tmp, &path).map_err(|e| io(e, &path))?;
        Ok(path)
    }

    async fn spawn(&self) -> Result<Worker, BackendError> {
        let dir = self.config.dir.clone();
        let mut cmd = match &self.launcher {
            Launcher::Python => {
                let mut cmd = Command::new(self.config.python_path());
                cmd.arg("-u").arg(self.worker_script()?).arg(&dir);
                cmd
            }
            Launcher::Program { path, env } => {
                let mut cmd = Command::new(path);
                cmd.arg(&dir).envs(env.iter().map(|(k, v)| (k, v)));
                cmd
            }
        };
        if dir.is_dir() {
            cmd.current_dir(&dir);
        }
        cmd.env("PYTHONUNBUFFERED", "1")
            .env("HF_HUB_OFFLINE", "1")
            .env("HF_HUB_CACHE", &self.hf_cache)
            .env("TQDM_DISABLE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        own_process_group(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| {
            BackendError::failed(
                "starting",
                "spawn_failed",
                format!("couldn't start the worker: {e}"),
                "check that the venv python runs (`<dir>/.venv/bin/python --version`)",
            )
        })?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let log = Arc::new(Mutex::new(VecDeque::with_capacity(LOG_TAIL_LINES)));
        let log_task = tokio::spawn(read_log(stderr, log.clone()));
        let mut worker = Worker {
            child,
            stdin,
            lines: BufReader::new(stdout).lines(),
            log,
            log_task: Some(log_task),
            hello: serde_json::Value::Null,
        };
        match tokio::time::timeout(self.hello_timeout, next_event(&mut worker.lines)).await {
            Ok(Some(ev)) if ev.event == "hello" && ev.protocol == PROTOCOL => {
                worker.hello = json!({
                    "protocol": ev.protocol,
                    "mlx": ev.mlx,
                    "python": ev.python,
                    "revision": ev.revision,
                });
                Ok(worker)
            }
            Ok(Some(ev)) if ev.event == "hello" => {
                kill_tree(&mut worker.child).await;
                Err(BackendError::failed(
                    "starting",
                    "worker_protocol",
                    format!(
                        "the worker speaks protocol {}, this server {PROTOCOL}",
                        ev.protocol
                    ),
                    "the worker file and the server come from different versions; restart the server",
                ))
            }
            Ok(Some(ev)) if ev.event == "error" => {
                // E.g. `missing_weights` at startup: the worker exits right after saying so.
                let _ = tokio::time::timeout(Duration::from_secs(5), worker.child.wait()).await;
                finish_log(&mut worker).await;
                kill_tree(&mut worker.child).await;
                Err(failure_from_event(ev, tail(&worker.log)))
            }
            Ok(_) => Err(self.died(&mut worker, "starting").await),
            Err(_) => {
                kill_tree(&mut worker.child).await;
                finish_log(&mut worker).await;
                Err(BackendError::failed(
                    "starting",
                    "worker_timeout",
                    format!(
                        "the worker didn't start within {} s",
                        self.hello_timeout.as_secs_f64().ceil() as u64
                    ),
                    "check that the stable-audio-3 venv imports MLX (run its `sa3` script once by hand)",
                )
                .failed_with_log(tail(&worker.log)))
            }
        }
    }

    /// The worker's stdout closed without an answer: report how it ended, with its log.
    async fn died(&self, worker: &mut Worker, stage: &str) -> BackendError {
        let status = tokio::time::timeout(Duration::from_secs(5), worker.child.wait()).await;
        let how = match status {
            Ok(Ok(s)) => describe_exit(&s),
            _ => {
                kill_tree(&mut worker.child).await;
                "stopped answering".to_string()
            }
        };
        finish_log(worker).await;
        let hint = if how.contains("SIGKILL") {
            "macOS may have killed it for memory; close heavy apps (Unreal Editor, Blender, other local models) and retry"
        } else {
            "see the worker log tail below; if it repeats, run the worker by hand (docs/worker-protocol.md)"
        };
        BackendError::failed(stage, "worker_crashed", format!("the worker {how}"), hint)
            .failed_with_log(tail(&worker.log))
    }

    fn schedule_idle_unload(&self) {
        if self.idle_unload_s == 0 {
            return;
        }
        let generation = self.idle_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let (slot, loaded, counter) = (
            self.slot.clone(),
            self.loaded.clone(),
            self.idle_generation.clone(),
        );
        let wait = Duration::from_secs(self.idle_unload_s);
        tokio::spawn(async move {
            tokio::time::sleep(wait).await;
            if counter.load(Ordering::SeqCst) != generation {
                return;
            }
            let mut guard = slot.lock().await;
            if counter.load(Ordering::SeqCst) == generation
                && let Some(worker) = guard.take()
            {
                loaded.lock().unwrap().take();
                stop(worker).await;
            }
        });
    }

    /// Run one request on `worker`. On any failure the worker is stopped (killed, for a cancel or a crash).
    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        worker: &mut Worker,
        id: &str,
        request: serde_json::Value,
        shape: &JobShape,
        model: &ModelSpec,
        progress: &ProgressSink,
        cancel: &CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError> {
        let sent = match worker.stdin.as_mut() {
            Some(stdin) => stdin
                .write_all(format!("{request}\n").as_bytes())
                .await
                .and(stdin.flush().await),
            None => Err(std::io::Error::other("worker stdin closed")),
        };
        if sent.is_err() {
            return Err(self.died(worker, "request").await);
        }
        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => return Err(BackendError::Cancelled),
                ev = next_event(&mut worker.lines) => ev,
            };
            let Some(ev) = event else {
                return Err(self.died(worker, "worker").await);
            };
            if !ev.id.is_empty() && ev.id != id {
                continue;
            }
            match ev.event.as_str() {
                "loading" => {
                    *self.loaded.lock().unwrap() = Some(model.id.clone());
                    progress.report(Stage::Loading);
                }
                "progress" => {
                    if let Some(stage) = map_stage(&ev.stage, ev.step, ev.total, shape.steps) {
                        progress.report(stage);
                    }
                }
                "result" => return read_result(ev, shape, &worker.hello).await,
                "error" => {
                    return Err(failure_from_event(ev, {
                        finish_log(worker).await;
                        tail(&worker.log)
                    }));
                }
                _ => {}
            }
        }
    }
}

/// What a job asked for, resolved: the numbers the result is checked against.
struct JobShape {
    seconds: f64,
    steps: u32,
    seeds: Vec<u64>,
}

#[async_trait::async_trait]
impl SfxBackend for Sa3MlxBackend {
    fn id(&self) -> &'static str {
        "sa3_mlx"
    }

    fn runs(&self) -> Runs {
        Runs::Local
    }

    async fn availability(&self, model: &ModelSpec) -> Availability {
        self.check(
            model,
            cfg!(all(target_os = "macos", target_arch = "aarch64")),
        )
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        request: &BackendRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError> {
        if let Availability::Unavailable { reason } = self.availability(model).await {
            return Err(BackendError::Unavailable { reason });
        }
        let BackendOptions::Sa3Mlx {
            steps: default_steps,
            ..
        } = &model.options
        else {
            return Err(BackendError::Unavailable {
                reason: format!("model `{}` has no sa3_mlx options", model.id),
            });
        };
        let shape = resolve_shape(request, *default_steps);
        let id = format!("g{:08x}", rand::random::<u32>());
        let message = json!({
            "op": "generate",
            "id": id,
            "prompt": request.prompt,
            "seconds": shape.seconds,
            "steps": shape.steps,
            "seeds": shape.seeds,
            "out_dir": request.out_dir,
        });

        // Invalidate any pending idle unload while this job runs.
        self.idle_generation.fetch_add(1, Ordering::SeqCst);
        let mut slot = self.slot.lock().await;
        let reuse = match slot.as_mut() {
            Some(w) => w.child.try_wait().ok().flatten().is_none(),
            None => false,
        };
        let mut worker = match (reuse, slot.take()) {
            (true, Some(w)) => w,
            (_, old) => {
                if let Some(old) = old {
                    stop(old).await;
                }
                self.loaded.lock().unwrap().take();
                progress.report(Stage::Loading);
                tokio::select! {
                    _ = cancel.cancelled() => return Err(BackendError::Cancelled),
                    w = self.spawn() => w?,
                }
            }
        };

        let outcome = self
            .run(&mut worker, &id, message, &shape, model, &progress, &cancel)
            .await;
        let healthy = outcome.is_ok() && worker.child.try_wait().ok().flatten().is_none();
        if healthy {
            *self.loaded.lock().unwrap() = Some(model.id.clone());
            *slot = Some(worker);
            drop(slot);
            self.schedule_idle_unload();
        } else {
            self.loaded.lock().unwrap().take();
            drop(slot);
            match &outcome {
                // A cancel or a crash: nothing to be polite to.
                Err(BackendError::Cancelled) => kill_tree(&mut worker.child).await,
                Err(BackendError::Failed(f)) if f.kind == "worker_crashed" => {
                    kill_tree(&mut worker.child).await
                }
                _ => stop(worker).await,
            }
        }
        clean_scratch(&request.out_dir);
        outcome
    }

    async fn unload(&self) {
        self.idle_generation.fetch_add(1, Ordering::SeqCst);
        let worker = self.slot.lock().await.take();
        self.loaded.lock().unwrap().take();
        if let Some(worker) = worker {
            stop(worker).await;
        }
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}

/// Seconds, steps and seeds the worker will be asked for.
///
/// A loop is asked for `d + loop_extra_s(d)` seconds; the job finisher crossfades it back to `d`. Seeds are
/// `[s, s+1, …]` with `s` random in `0..2^31` when the caller gave none (distinct, as the worker requires).
fn resolve_shape(request: &BackendRequest, default_steps: u32) -> JobShape {
    let d = request.duration_s.unwrap_or(DEFAULT_DURATION_S);
    let seconds = if request.loop_ {
        d + crate::audio::edit::loop_extra_s(d)
    } else {
        d
    };
    let first = request
        .seed
        .unwrap_or_else(|| u64::from(rand::random::<u32>() & 0x7fff_ffff));
    JobShape {
        seconds,
        steps: request.steps.unwrap_or(default_steps),
        seeds: (0..u64::from(request.variations.max(1)))
            .map(|i| first.wrapping_add(i))
            .collect(),
    }
}

/// Turn a `result` event into clips (reading and deleting the worker's raw files).
async fn read_result(
    ev: Event,
    shape: &JobShape,
    hello: &serde_json::Value,
) -> Result<Vec<RawClip>, BackendError> {
    let bad = |message: String| {
        BackendError::failed(
            "write",
            "bad_output",
            message,
            "report this: the worker's result doesn't match its protocol",
        )
    };
    if ev.files.len() != shape.seeds.len() {
        return Err(bad(format!(
            "the worker returned {} file(s) for {} seed(s)",
            ev.files.len(),
            shape.seeds.len()
        )));
    }
    let seeds = if ev.seeds.len() == ev.files.len() {
        ev.seeds.clone()
    } else {
        shape.seeds.clone()
    };
    let size = ev.files.len();
    let mut clips = Vec::with_capacity(size);
    for (index, (path, seed)) in ev.files.iter().zip(&seeds).enumerate() {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| bad(format!("couldn't read {}: {e}", path.display())))?;
        let _ = tokio::fs::remove_file(path).await;
        let clip = Clip::from_f32le_bytes(&bytes, ev.sample_rate, ev.channels)
            .map_err(|e| bad(format!("{}: {e}", path.display())))?;
        if ev.frames > 0 && clip.frames() as u64 != ev.frames {
            return Err(bad(format!(
                "{} holds {} frames, the worker said {}",
                path.display(),
                clip.frames(),
                ev.frames
            )));
        }
        clips.push(RawClip {
            samples: clip.samples,
            sample_rate: ev.sample_rate,
            channels: ev.channels,
            seed: Some(*seed),
            source_format: "f32".into(),
            credits: None,
            details: json!({
                "batch": { "size": size, "index": index },
                "steps": shape.steps,
                "seconds_requested": shape.seconds,
                "timings": ev.timings,
                "peak_mlx_mb": ev.peak_mlx_mb,
                "active_mlx_mb": ev.active_mlx_mb,
                "peak_rss_mb": ev.peak_rss_mb,
                "worker": hello,
            }),
        });
    }
    Ok(clips)
}

/// Delete what a job leaves in its scratch folder: result files and half-written `.partial` files (a killed worker can
/// leave them). The job manager removes the folder itself afterwards.
fn clean_scratch(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".partial") || name.ends_with(".f32") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// An explained error from the worker (R8), keeping stage, kind and the worker's hint, with the log tail attached.
fn failure_from_event(ev: Event, log_tail: String) -> BackendError {
    let stage = if ev.stage.is_empty() {
        "worker".to_string()
    } else {
        ev.stage.clone()
    };
    let (kind, hint) = match ev.kind.as_str() {
        "out_of_memory" => (
            "out_of_memory",
            if ev.hint.is_empty() {
                OOM_HINT.to_string()
            } else {
                ev.hint.clone()
            },
        ),
        "missing_weights" => (
            "missing_weights",
            "the weights aren't where the worker looks (<dir>/models/mlx); `list_models` shows this model's status with the \
             exact download command"
                .to_string(),
        ),
        "nan_output" => (
            "nan_output",
            if ev.hint.is_empty() {
                "try another seed or a different prompt".to_string()
            } else {
                ev.hint.clone()
            },
        ),
        "bad_request" => (
            "bad_request",
            "the worker rejected a request the server built; report this as a bug".to_string(),
        ),
        other => (
            if other.is_empty() {
                "worker_exception"
            } else {
                other
            },
            ev.hint.clone(),
        ),
    };
    BackendError::failed(&stage, kind, ev.message, hint).failed_with_log(log_tail)
}

/// Ask a kept worker to unload and exit; kill it if it doesn't within a few seconds.
async fn stop(mut worker: Worker) {
    if let Some(stdin) = worker.stdin.as_mut() {
        let _ = stdin
            .write_all(b"{\"op\":\"unload\",\"id\":\"unload\"}\n")
            .await;
        let _ = stdin.flush().await;
    }
    worker.stdin.take();
    if tokio::time::timeout(EXIT_TIMEOUT, worker.child.wait())
        .await
        .is_err()
    {
        kill_tree(&mut worker.child).await;
    }
}

/// Next protocol event; non-JSON lines on stdout are skipped. `None` at EOF.
async fn next_event(lines: &mut Lines<BufReader<ChildStdout>>) -> Option<Event> {
    loop {
        let line = lines.next_line().await.ok()??;
        if let Ok(ev) = serde_json::from_str::<Event>(line.trim()) {
            return Some(ev);
        }
    }
}

/// Wait (briefly) for the stderr reader to reach EOF so the tail includes the worker's last words.
async fn finish_log(worker: &mut Worker) {
    if let Some(task) = worker.log_task.take() {
        let _ = tokio::time::timeout(Duration::from_millis(500), task).await;
    }
}

/// Keep the last lines of the worker's stderr (splitting `\r` redraws).
async fn read_log(mut stderr: tokio::process::ChildStderr, log: Arc<Mutex<VecDeque<String>>>) {
    let mut buf = vec![0u8; 8192];
    let mut pending = String::new();
    loop {
        let n = match stderr.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        pending.push_str(&String::from_utf8_lossy(&buf[..n]));
        while let Some(pos) = pending.find(['\n', '\r']) {
            let line: String = pending.drain(..=pos).collect();
            push_line(&log, line.trim_end());
        }
        if pending.len() > 4096 {
            let line = std::mem::take(&mut pending);
            push_line(&log, &line);
        }
    }
    if !pending.is_empty() {
        push_line(&log, pending.trim_end());
    }
}

fn push_line(log: &Mutex<VecDeque<String>>, line: &str) {
    if line.trim().is_empty() {
        return;
    }
    let mut log = log.lock().unwrap();
    if log.len() == LOG_TAIL_LINES {
        log.pop_front();
    }
    log.push_back(line.chars().take(500).collect());
}

fn tail(log: &Mutex<VecDeque<String>>) -> String {
    log.lock()
        .unwrap()
        .iter()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n")
}

/// Worker stages as job progress. Sampler steps are the count that moves; the condition stage sits at 0 and decoding at
/// "all steps done", so the number only ever goes up.
fn map_stage(stage: &str, step: u32, total: u32, steps: u32) -> Option<Stage> {
    Some(match stage {
        "condition" => Stage::Generating { n: 0, of: steps },
        "sample" => Stage::Generating {
            n: step,
            of: total.max(1),
        },
        "decode" => Stage::Generating {
            n: steps,
            of: steps,
        },
        _ => return None,
    })
}

fn describe_exit(status: &std::process::ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            let name = match sig {
                9 => "SIGKILL",
                6 => "SIGABRT",
                11 => "SIGSEGV",
                15 => "SIGTERM",
                _ => "a signal",
            };
            return format!("was killed by {name} ({sig})");
        }
    }
    match status.code() {
        Some(code) => format!("exited with code {code} before answering"),
        None => "exited before answering".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;
    use crate::registry::ModelRegistry;

    fn model() -> ModelSpec {
        ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("sa3-small-sfx")
            .unwrap()
            .clone()
    }

    fn backend(dir: &Path, hf: &Path) -> Sa3MlxBackend {
        Sa3MlxBackend::new(
            Sa3MlxConfig {
                dir: dir.to_path_buf(),
                licence_accepted: None,
            },
            hf.to_path_buf(),
            std::env::temp_dir(),
            0,
        )
    }

    #[test]
    fn availability_rules() {
        let reason = |a: Availability| a.reason().unwrap_or("ready").to_string();
        let dir = tempfile::tempdir().unwrap();
        let hf = tempfile::tempdir().unwrap();
        let b = backend(dir.path(), hf.path());
        assert!(reason(b.check(&model(), false)).contains("needs Apple Silicon"));
        assert!(reason(b.check(&model(), true)).contains("no venv python"));
        let python = crate::config::venv_python(dir.path());
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        // Weights neither downloaded nor linked: the reason carries the download command.
        let r = reason(b.check(&model(), true));
        assert!(
            r.contains("dit_sm-sfx_f16.npz") && r.contains("hf download"),
            "{r}"
        );
        let folder = dir.path().join("models/mlx");
        std::fs::create_dir_all(&folder).unwrap();
        for name in ["dit_sm-sfx_f16.npz", "t5gemma_f16.npz"] {
            std::fs::write(folder.join(name), b"x").unwrap();
        }
        // A dangling link counts as missing, and names only what's missing.
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            dir.path().join("nowhere.npz"),
            folder.join("same_s_decoder_f32.npz"),
        )
        .unwrap();
        let r = reason(b.check(&model(), true));
        assert!(
            r.starts_with("missing weight file(s) same_s_decoder_f32.npz under"),
            "{r}"
        );
        #[cfg(unix)]
        std::fs::remove_file(folder.join("same_s_decoder_f32.npz")).unwrap();
        std::fs::write(folder.join("same_s_decoder_f32.npz"), b"x").unwrap();
        assert_eq!(b.check(&model(), true), Availability::Ready);
    }

    #[test]
    fn a_stand_in_program_only_needs_to_exist() {
        let b = Sa3MlxBackend::with_launcher(
            Sa3MlxConfig {
                dir: "/nowhere".into(),
                licence_accepted: None,
            },
            "/nowhere".into(),
            std::env::temp_dir(),
            0,
            Launcher::Program {
                path: "/definitely/not/here".into(),
                env: vec![],
            },
        );
        assert!(
            b.check(&model(), false)
                .reason()
                .unwrap()
                .contains("worker program")
        );
    }

    #[test]
    fn worker_script_is_written_once_per_version() {
        let cache = tempfile::tempdir().unwrap();
        let b = Sa3MlxBackend::new(
            Sa3MlxConfig {
                dir: "/x".into(),
                licence_accepted: None,
            },
            "/y".into(),
            cache.path().into(),
            0,
        );
        let path = b.worker_script().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), WORKER_SOURCE);
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("sa3_mlx_worker-")
        );
        assert_eq!(b.worker_script().unwrap(), path);
    }

    #[test]
    fn stage_names_map_and_only_count_up() {
        assert_eq!(
            map_stage("condition", 0, 1, 8),
            Some(Stage::Generating { n: 0, of: 8 })
        );
        assert_eq!(
            map_stage("sample", 3, 8, 8),
            Some(Stage::Generating { n: 3, of: 8 })
        );
        assert_eq!(
            map_stage("decode", 1, 4, 8),
            Some(Stage::Generating { n: 8, of: 8 })
        );
        assert_eq!(map_stage("write", 0, 0, 8), None);
    }

    #[test]
    fn log_keeps_a_bounded_tail() {
        let log = Mutex::new(VecDeque::new());
        for i in 0..100 {
            push_line(&log, &format!("line {i}"));
        }
        assert_eq!(log.lock().unwrap().len(), LOG_TAIL_LINES);
        assert!(tail(&log).ends_with("line 99"));
        assert!(tail(&log).starts_with("line 60"));
    }

    #[test]
    fn the_embedded_worker_speaks_this_protocol() {
        assert!(WORKER_SOURCE.contains(&format!("PROTOCOL = {PROTOCOL}")));
        assert!(WORKER_SOURCE.contains("missing_weights"));
    }

    #[test]
    fn seeds_are_consecutive_and_distinct_and_loops_ask_for_extra_seconds() {
        let mut r = BackendRequest {
            duration_s: Some(1.0),
            variations: 3,
            seed: Some(100),
            ..Default::default()
        };
        let shape = resolve_shape(&r, 8);
        assert_eq!(shape.seeds, [100, 101, 102]);
        assert_eq!((shape.seconds, shape.steps), (1.0, 8));
        r.seed = None;
        r.loop_ = true;
        r.steps = Some(4);
        let shape = resolve_shape(&r, 8);
        assert!(shape.seeds[0] < (1 << 31) && shape.seeds[1] == shape.seeds[0] + 1);
        assert!(
            (shape.seconds - (1.0 + crate::audio::edit::loop_extra_s(1.0))).abs() < 1e-9,
            "{}",
            shape.seconds
        );
        assert_eq!(shape.steps, 4);
        r.duration_s = None;
        r.loop_ = false;
        assert_eq!(resolve_shape(&r, 8).seconds, DEFAULT_DURATION_S);
    }

    #[test]
    fn worker_errors_keep_stage_kind_and_get_a_useful_hint() {
        let ev = |kind: &str, hint: &str| Event {
            event: "error".into(),
            stage: "decode".into(),
            kind: kind.into(),
            message: "boom".into(),
            hint: hint.into(),
            ..Default::default()
        };
        let BackendError::Failed(f) = failure_from_event(ev("out_of_memory", ""), "tail".into())
        else {
            panic!()
        };
        assert_eq!(
            (f.stage.as_str(), f.kind.as_str()),
            ("decode", "out_of_memory")
        );
        assert!(f.hint.contains("Unreal"), "{}", f.hint);
        assert_eq!(f.log_tail, "tail");
        let BackendError::Failed(f) = failure_from_event(ev("missing_weights", ""), String::new())
        else {
            panic!()
        };
        assert!(f.hint.contains("list_models"));
        let BackendError::Failed(f) =
            failure_from_event(ev("worker_exception", "fake"), String::new())
        else {
            panic!()
        };
        assert_eq!(
            (f.kind.as_str(), f.hint.as_str()),
            ("worker_exception", "fake")
        );
    }
}
