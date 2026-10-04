//! The sa3_mlx backend against `fake-worker` (the real spawn, protocol, cancel and keep-alive code paths).
#![cfg(all(feature = "mock", feature = "sa3_mlx"))]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mcp_sfxgen::backend::sa3_mlx::{Launcher, Sa3MlxBackend};
use mcp_sfxgen::backend::{BackendError, BackendRequest, ProgressSink, SfxBackend};
use mcp_sfxgen::config::{MODELS_EXAMPLE, Sa3MlxConfig};
use mcp_sfxgen::registry::{ModelRegistry, ModelSpec};
use tokio_util::sync::CancellationToken;

struct Fixture {
    dir: tempfile::TempDir,
    backend: Sa3MlxBackend,
}

impl Fixture {
    fn out_dir(&self) -> PathBuf {
        self.dir.path().join("out")
    }

    fn files_in_out(&self) -> Vec<String> {
        std::fs::read_dir(self.out_dir())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    }

    fn requests(&self) -> Vec<serde_json::Value> {
        std::fs::read_to_string(self.dir.path().join("requests.log"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

fn fixture(mode: &str, idle_unload_s: u64) -> Fixture {
    fixture_with(mode, idle_unload_s, Duration::from_secs(60), &[])
}

fn fixture_with(
    mode: &str,
    idle_unload_s: u64,
    hello_timeout: Duration,
    extra_env: &[(&str, &str)],
) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("out")).unwrap();
    let mut env = vec![
        ("FAKE_WORKER_MODE".to_string(), mode.to_string()),
        (
            "FAKE_WORKER_LOG".to_string(),
            dir.path().join("requests.log").display().to_string(),
        ),
    ];
    env.extend(
        extra_env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string())),
    );
    let backend = Sa3MlxBackend::with_launcher(
        Sa3MlxConfig {
            dir: dir.path().join("sa3"),
            licence_accepted: None,
        },
        dir.path().join("hub"),
        dir.path().join("cache"),
        idle_unload_s,
        Launcher::Program {
            path: PathBuf::from(env!("CARGO_BIN_EXE_fake-worker")),
            env,
        },
    )
    .with_hello_timeout(hello_timeout);
    Fixture { dir, backend }
}

fn model() -> ModelSpec {
    ModelRegistry::from_toml(MODELS_EXAMPLE)
        .unwrap()
        .get("sa3-small-sfx")
        .unwrap()
        .clone()
}

fn request(out_dir: &Path, variations: u32) -> BackendRequest {
    BackendRequest {
        prompt: "a cartoon meep".into(),
        duration_s: Some(1.0),
        variations,
        seed: Some(101),
        sample_rate: 44_100,
        out_dir: out_dir.to_path_buf(),
        ..Default::default()
    }
}

fn recorder() -> (ProgressSink, Arc<Mutex<Vec<String>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let s = seen.clone();
    (
        ProgressSink::new(move |stage| s.lock().unwrap().push(stage.to_string())),
        seen,
    )
}

async fn failure(mode: &str) -> (Fixture, BackendError) {
    let f = fixture(mode, 0);
    let err = f
        .backend
        .generate(
            &model(),
            &request(&f.out_dir(), 2),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    (f, err)
}

fn failed(err: BackendError) -> mcp_sfxgen::backend::BackendFailure {
    match err {
        BackendError::Failed(f) => f,
        other => panic!("expected an explained failure, got {other:?}"),
    }
}

#[tokio::test]
async fn success_returns_batched_clips_and_reports_stages() {
    let f = fixture("ok", 0);
    let (progress, seen) = recorder();
    let clips = f
        .backend
        .generate(
            &model(),
            &request(&f.out_dir(), 2),
            progress,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(clips.len(), 2);
    for (i, clip) in clips.iter().enumerate() {
        assert_eq!(clip.seed, Some(101 + i as u64), "consecutive seeds");
        assert_eq!((clip.sample_rate, clip.channels), (44_100, 2));
        assert_eq!(clip.frames(), 44_100);
        assert_eq!(clip.source_format, "f32");
        assert_eq!(clip.credits, None);
        assert!((clip.samples[100].abs() - clip.samples[101].abs()).abs() < 1e-6);
        assert_eq!(clip.details["batch"]["size"], 2);
        assert_eq!(clip.details["batch"]["index"], i);
        assert_eq!(clip.details["worker"]["protocol"], 1);
        assert_eq!(clip.details["worker"]["mlx"], "0.32.2");
        assert_eq!(clip.details["peak_mlx_mb"], 2400.0);
        assert_eq!(clip.details["steps"], 8);
    }
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen[0], "loading model", "{seen:?}");
    for stage in [
        "generating 0 of 8",
        "generating 3 of 8",
        "generating 8 of 8",
    ] {
        assert!(
            seen.iter().any(|s| s == stage),
            "{stage} missing from {seen:?}"
        );
    }
    assert!(
        f.files_in_out().is_empty(),
        "raw files are read and deleted: {:?}",
        f.files_in_out()
    );
    assert_eq!(f.backend.loaded_model().as_deref(), Some("sa3-small-sfx"));
    // The request carried the resolved parameters.
    let req = &f.requests()[0];
    assert_eq!(req["op"], "generate");
    assert_eq!(req["prompt"], "a cartoon meep");
    // 1 s is under the model's 2 s minimum: it is generated at 2 s and each clip is cut back to its event.
    assert_eq!(req["seconds"], 2.0);
    assert_eq!(req["steps"], 8);
    assert_eq!(req["seeds"], serde_json::json!([101, 102]));
    assert_eq!(req["out_dir"], f.out_dir().display().to_string());
    assert!(
        req["id"]
            .as_str()
            .unwrap()
            .chars()
            .all(|c| c.is_ascii_alphanumeric())
    );
}

#[tokio::test]
async fn a_loop_asks_for_the_extra_seconds_and_default_duration_is_two() {
    let f = fixture("ok", 0);
    let mut req = request(&f.out_dir(), 1);
    req.loop_ = true;
    req.steps = Some(4);
    let clips = f
        .backend
        .generate(
            &model(),
            &req,
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    // A 1 s loop needs 1 s + extra; that's under the 2 s minimum, so it is generated at 2 s and cut to 1 s + extra.
    let extra = mcp_sfxgen::audio::edit::loop_extra_s(1.0);
    let sent = &f.requests()[0];
    assert!((sent["seconds"].as_f64().unwrap() - 2.0).abs() < 1e-9);
    assert_eq!(sent["steps"], 4);
    assert!((clips[0].duration_s() - (1.0 + extra)).abs() < 1e-3);

    req.loop_ = false;
    req.duration_s = None;
    req.seed = None;
    f.backend
        .generate(
            &model(),
            &req,
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    let sent = &f.requests()[1];
    assert_eq!(sent["seconds"], 2.0);
    let seed = sent["seeds"][0].as_u64().unwrap();
    assert!(seed < (1 << 31), "a random seed is below 2^31: {seed}");
}

#[tokio::test]
async fn explained_errors_keep_stage_kind_hint_and_log() {
    let (_f, err) = failure("error:sample:nan_output").await;
    let f = failed(err);
    assert_eq!(
        (f.stage.as_str(), f.kind.as_str(), f.hint.as_str()),
        ("sample", "nan_output", "fake hint")
    );
    assert!(
        f.log_tail.contains("fake sa3 worker ready"),
        "{}",
        f.log_tail
    );
}

#[tokio::test]
async fn out_of_memory_keeps_the_hint_and_maps_to_its_code() {
    let (f, err) = failure("error:decode:out_of_memory").await;
    let failure = failed(err);
    assert_eq!(failure.kind, "out_of_memory");
    assert_eq!(failure.stage, "decode");
    assert_eq!(
        failure.hint, "fake hint",
        "the worker's own hint is kept (the unit test covers the fallback that names Unreal)"
    );
    // The server-level mapping gives the stable code.
    assert_eq!(
        BackendError::Failed(failure).into_error("sa3_mlx").code(),
        "out_of_memory"
    );
    assert_eq!(
        f.backend.loaded_model(),
        None,
        "a failed job stops the worker"
    );
}

#[tokio::test]
async fn missing_weights_while_loading_points_at_the_model_status() {
    let (_f, err) = failure("error:loading:missing_weights").await;
    let f = failed(err);
    assert_eq!(
        (f.stage.as_str(), f.kind.as_str()),
        ("loading", "missing_weights")
    );
    assert!(f.hint.contains("list_models"), "{}", f.hint);
}

#[tokio::test]
async fn missing_weights_at_startup_is_reported_with_the_log() {
    let (_f, err) = failure("missing_weights").await;
    let f = failed(err);
    assert_eq!(
        (f.stage.as_str(), f.kind.as_str()),
        ("startup", "missing_weights")
    );
    assert!(f.message.contains("t5gemma_f16.npz"), "{}", f.message);
    assert!(f.hint.contains("list_models"));
}

#[tokio::test]
async fn a_worker_exception_passes_through() {
    let (_f, err) = failure("error:condition:worker_exception").await;
    let f = failed(err);
    assert_eq!(
        (f.stage.as_str(), f.kind.as_str()),
        ("condition", "worker_exception")
    );
}

#[tokio::test]
async fn a_crash_is_reported_with_its_exit_and_the_log_tail() {
    let (f, err) = failure("crash:sample").await;
    let failure = failed(err);
    assert_eq!(failure.kind, "worker_crashed");
    assert!(
        failure.message.contains("exited with code 3"),
        "{}",
        failure.message
    );
    assert!(
        failure.log_tail.contains("fake crash in sample"),
        "{}",
        failure.log_tail
    );
    assert_eq!(
        BackendError::Failed(failure).into_error("sa3_mlx").code(),
        "worker_crashed"
    );
    assert_eq!(f.backend.loaded_model(), None);
}

#[tokio::test]
async fn a_crash_while_writing_leaves_no_partial_files() {
    let (f, err) = failure("crash:write").await;
    assert_eq!(failed(err).kind, "worker_crashed");
    assert!(
        f.files_in_out().is_empty(),
        "the half-written .partial file is deleted: {:?}",
        f.files_in_out()
    );
}

#[tokio::test]
async fn a_protocol_mismatch_is_caught_at_hello() {
    let (_f, err) = failure("bad_protocol").await;
    assert_eq!(failed(err).kind, "worker_protocol");
}

#[tokio::test]
async fn a_silent_worker_times_out_at_hello() {
    let f = fixture_with("hang_hello", 0, Duration::from_millis(600), &[]);
    let started = Instant::now();
    let err = f
        .backend
        .generate(
            &model(),
            &request(&f.out_dir(), 1),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    let failure = failed(err);
    assert_eq!(failure.kind, "worker_timeout");
    assert!(
        failure.message.contains("didn't start"),
        "{}",
        failure.message
    );
    assert!(started.elapsed() < Duration::from_secs(10));
    assert_eq!(f.backend.loaded_model(), None);
}

#[tokio::test]
async fn cancel_stops_the_worker_mid_stage_and_leaves_no_files() {
    let f = fixture("slow", 0);
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    let (progress, seen) = recorder();
    let seen2 = seen.clone();
    tokio::spawn(async move {
        loop {
            if seen2
                .lock()
                .unwrap()
                .iter()
                .any(|s| s.starts_with("generating 2"))
            {
                c.cancel();
                return;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    });
    let started = Instant::now();
    let err = f
        .backend
        .generate(&model(), &request(&f.out_dir(), 2), progress, cancel)
        .await
        .unwrap_err();
    assert_eq!(err, BackendError::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(10));
    assert!(f.files_in_out().is_empty(), "{:?}", f.files_in_out());
    assert_eq!(f.backend.loaded_model(), None);
}

#[tokio::test]
async fn cancel_during_decode_deletes_files_the_worker_was_writing() {
    // The worker is killed while it works; whatever it had on disk is removed.
    let f = fixture_with(
        "slow",
        0,
        Duration::from_secs(60),
        &[("FAKE_WORKER_STEP_MS", "150")],
    );
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1400)).await;
        c.cancel();
    });
    let err = f
        .backend
        .generate(
            &model(),
            &request(&f.out_dir(), 4),
            ProgressSink::noop(),
            cancel,
        )
        .await;
    assert!(
        matches!(err, Err(BackendError::Cancelled)) || err.is_ok(),
        "{err:?}"
    );
    assert!(f.files_in_out().is_empty(), "{:?}", f.files_in_out());
}

#[tokio::test]
async fn cancel_while_the_worker_starts_up() {
    let f = fixture("hang_hello", 0);
    let cancel = CancellationToken::new();
    let c = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        c.cancel();
    });
    let started = Instant::now();
    let err = f
        .backend
        .generate(
            &model(),
            &request(&f.out_dir(), 1),
            ProgressSink::noop(),
            cancel,
        )
        .await
        .unwrap_err();
    assert_eq!(err, BackendError::Cancelled);
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn the_worker_is_reused_until_unload() {
    let f = fixture("ok", 0);
    let run = |seed: u64| {
        let mut req = request(&f.out_dir(), 1);
        req.seed = Some(seed);
        let backend = &f.backend;
        async move {
            let (progress, seen) = recorder();
            backend
                .generate(&model(), &req, progress, CancellationToken::new())
                .await
                .unwrap();
            seen.lock().unwrap().clone()
        }
    };
    let first = run(1).await;
    assert!(first.iter().any(|s| s == "loading model"));
    let second = run(2).await;
    assert!(
        !second.iter().any(|s| s == "loading model"),
        "the second job ran in the same, already loaded worker: {second:?}"
    );
    assert_eq!(f.backend.loaded_model().as_deref(), Some("sa3-small-sfx"));
    f.backend.unload().await;
    assert_eq!(f.backend.loaded_model(), None);
    let sent = f.requests();
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert_eq!(sent.last().unwrap()["op"], "unload");
    f.backend.unload().await; // idempotent
}

#[tokio::test]
async fn a_kept_worker_unloads_when_idle() {
    let f = fixture("ok", 1);
    f.backend
        .generate(
            &model(),
            &request(&f.out_dir(), 1),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(f.backend.loaded_model().is_some());
    tokio::time::sleep(Duration::from_millis(1800)).await;
    assert_eq!(f.backend.loaded_model(), None);
    assert_eq!(f.requests().last().unwrap()["op"], "unload");
}

#[tokio::test]
async fn a_new_job_cancels_the_pending_idle_unload() {
    let f = fixture("ok", 1);
    for seed in [1, 2] {
        let mut req = request(&f.out_dir(), 1);
        req.seed = Some(seed);
        f.backend
            .generate(
                &model(),
                &req,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(600)).await;
    }
    // 1.2 s since the first job but only 0.6 s since the second: still loaded.
    assert!(f.backend.loaded_model().is_some());
}

#[tokio::test]
async fn after_unload_the_next_job_starts_a_new_worker() {
    let f = fixture("ok", 0);
    f.backend
        .generate(
            &model(),
            &request(&f.out_dir(), 1),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap();
    // After an unload the next job starts a new worker.
    f.backend.unload().await;
    let (progress, seen) = recorder();
    f.backend
        .generate(
            &model(),
            &request(&f.out_dir(), 1),
            progress,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert!(seen.lock().unwrap().iter().any(|s| s == "loading model"));
}

#[tokio::test]
async fn an_unavailable_program_is_reported_before_anything_runs() {
    let dir = tempfile::tempdir().unwrap();
    let backend = Sa3MlxBackend::with_launcher(
        Sa3MlxConfig {
            dir: dir.path().to_path_buf(),
            licence_accepted: None,
        },
        dir.path().join("hub"),
        dir.path().join("cache"),
        0,
        Launcher::Program {
            path: dir.path().join("no-such-worker"),
            env: vec![],
        },
    );
    let availability = backend.availability(&model()).await;
    assert!(availability.reason().unwrap().contains("not found"));
    let err = backend
        .generate(
            &model(),
            &request(dir.path(), 1),
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .unwrap_err();
    assert!(matches!(err, BackendError::Unavailable { .. }), "{err:?}");
}
