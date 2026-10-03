//! Deterministic fake backend for tests and demos (feature `mock`).
//!
//! Makes a tone (even seed) or a noise burst (odd seed) per clip, walks the real stage sequence with `step_delay`
//! per step, honours cancellation, and can be told to fail the next job at a stage with a given kind. It runs in
//! either lane: `MockBackend::local` (id `mock`) and `MockBackend::cloud` (id `mock-cloud`, no seeds, reports credits
//! like ElevenLabs: 10 per second, 50 when it picks the length, and a loop comes back 0.25 s long).
//!
//! The server registers both under the `mock` feature, configured by `MCP_SFXGEN_MOCK_DELAY_MS` (per step, default 50)
//! and `MCP_SFXGEN_MOCK_FAIL` (`stage:kind`, e.g. `generating:worker_crashed`, fails the first job).

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{
    Availability, BackendError, BackendRequest, ProgressSink, RawClip, Runs, SfxBackend, Stage,
};
use crate::registry::ModelSpec;

/// A local and a cloud mock model for tests (`mock` and `mock-cloud` backends).
pub const MOCK_MODELS: &str = r#"
[[model]]
id = "mock-local"
backend = "mock"
duration_s = { min = 0.5, max = 10.0 }
memory_mb = 100
typical_s = 0.5
options = { kind = "mock", runs = "local" }
licence = [{ component = "mock", name = "MIT", commercial = "yes" }]

[[model]]
id = "mock-cloud"
backend = "mock-cloud"
duration_s = { min = 0.5, max = 10.0 }
native_loop = true
typical_s = 0.5
options = { kind = "mock", runs = "cloud" }
licence = [{ component = "mock", name = "Mock terms", commercial = "conditional", note = "paid plans only" }]
privacy = "the prompt is sent to the mock cloud"
"#;

/// Default length when the request names none (local), or the cloud's "model picks" length.
pub const MOCK_DEFAULT_S: f64 = 2.0;
pub const MOCK_AUTO_S: f64 = 1.5;

pub struct MockBackend {
    id: &'static str,
    runs: Runs,
    step_delay: Duration,
    availability: Mutex<Availability>,
    loaded: Mutex<Option<String>>,
    fail_next: Mutex<Option<(String, String)>>,
    /// Finished `generate` calls.
    pub generations: AtomicUsize,
    pub unloads: AtomicUsize,
    /// `generate` calls in flight now, and the most there ever were (lane-parallelism checks).
    pub active: AtomicUsize,
    pub max_active: AtomicUsize,
}

impl MockBackend {
    pub fn new(id: &'static str, runs: Runs, step_delay: Duration) -> Self {
        Self {
            id,
            runs,
            step_delay,
            availability: Mutex::new(Availability::Ready),
            loaded: Mutex::new(None),
            fail_next: Mutex::new(None),
            generations: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        }
    }

    /// The local mock, backend id `mock`.
    pub fn local(step_delay: Duration) -> Self {
        Self::new("mock", Runs::Local, step_delay)
    }

    /// The cloud mock, backend id `mock-cloud`.
    pub fn cloud(step_delay: Duration) -> Self {
        Self::new("mock-cloud", Runs::Cloud, step_delay)
    }

    /// Delay and failure injection from the environment (see the module docs).
    pub fn with_env(self) -> Self {
        if let Ok(spec) = std::env::var("MCP_SFXGEN_MOCK_FAIL")
            && let Some((stage, kind)) = spec.split_once(':')
        {
            self.fail_next(stage, kind);
        }
        self
    }

    /// The step delay from `MCP_SFXGEN_MOCK_DELAY_MS`, default 50 ms.
    pub fn env_delay() -> Duration {
        Duration::from_millis(
            std::env::var("MCP_SFXGEN_MOCK_DELAY_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
        )
    }

    /// Fail the next job during `stage` (`loading` or `generating`) with `kind`.
    pub fn fail_next(&self, stage: &str, kind: &str) {
        *self.fail_next.lock().unwrap() = Some((stage.to_string(), kind.to_string()));
    }

    pub fn set_availability(&self, availability: Availability) {
        *self.availability.lock().unwrap() = availability;
    }

    fn check_failure(&self, stage: &str) -> Result<(), BackendError> {
        let mut fail = self.fail_next.lock().unwrap();
        if fail.as_ref().is_some_and(|(s, _)| s == stage) {
            let (stage, kind) = fail.take().unwrap();
            return Err(BackendError::failed(
                &stage,
                &kind,
                format!("injected {kind} failure"),
                "this is the mock backend",
            ));
        }
        Ok(())
    }

    async fn tick(&self, cancel: &CancellationToken) -> Result<(), BackendError> {
        tokio::select! {
            _ = cancel.cancelled() => Err(BackendError::Cancelled),
            _ = tokio::time::sleep(self.step_delay) => Ok(()),
        }
    }

    /// The clip for `seed`: 0.5 amplitude tone at `220 + seed % 880` Hz for an even seed, noise for an odd one.
    pub fn render(seed: u64, seconds: f64, sample_rate: u32) -> Vec<f32> {
        let frames = (seconds * f64::from(sample_rate)).round() as usize;
        let fade = (f64::from(sample_rate) * 0.005) as usize;
        let freq = 220.0 + (seed % 880) as f64;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut out = Vec::with_capacity(frames * 2);
        for i in 0..frames {
            let envelope = if fade == 0 {
                1.0
            } else {
                ((i + 1).min(fade).min(frames - i) as f64 / fade as f64).min(1.0)
            };
            let value = if seed.is_multiple_of(2) {
                0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / f64::from(sample_rate)).sin()
            } else {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                0.3 * ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
            };
            let v = (value * envelope) as f32;
            out.push(v);
            out.push(v);
        }
        out
    }
}

/// FNV-1a: a stable stand-in "seed" for a cloud mock, which has none to report.
fn hash(prompt: &str, index: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in prompt.bytes().chain(index.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[async_trait::async_trait]
impl SfxBackend for MockBackend {
    fn id(&self) -> &'static str {
        self.id
    }

    fn runs(&self) -> Runs {
        self.runs
    }

    async fn availability(&self, _model: &ModelSpec) -> Availability {
        self.availability.lock().unwrap().clone()
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        request: &BackendRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError> {
        let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(now, Ordering::SeqCst);
        let outcome = self.run(model, request, progress, &cancel).await;
        self.active.fetch_sub(1, Ordering::SeqCst);
        if outcome.is_ok() {
            self.generations.fetch_add(1, Ordering::SeqCst);
        }
        outcome
    }

    async fn unload(&self) {
        if self.loaded.lock().unwrap().take().is_some() {
            self.unloads.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }

    fn provider(&self) -> Option<&'static str> {
        (self.runs == Runs::Cloud).then_some("mock-cloud")
    }
}

impl MockBackend {
    async fn run(
        &self,
        model: &ModelSpec,
        request: &BackendRequest,
        progress: ProgressSink,
        cancel: &CancellationToken,
    ) -> Result<Vec<RawClip>, BackendError> {
        progress.report(Stage::Loading);
        if self.runs == Runs::Local {
            *self.loaded.lock().unwrap() = Some(model.id.clone());
        }
        self.tick(cancel).await?;
        self.check_failure("loading")?;

        let cloud = self.runs == Runs::Cloud;
        // Like the real backends: a local model always delivers 44.1 kHz, a cloud one can deliver 48 kHz when asked.
        let rate = if cloud && request.sample_rate != 0 {
            request.sample_rate
        } else {
            44_100
        };
        let n = request.variations.max(1);
        let first_seed = request
            .seed
            .unwrap_or_else(|| rand::random::<u32>() as u64 & 0x7fff_ffff);
        let mut clips = Vec::new();
        for i in 0..n {
            progress.report(Stage::Generating { n: i + 1, of: n });
            self.check_failure("generating")?;
            self.tick(cancel).await?;
            let seed = first_seed + u64::from(i);
            let mut seconds =
                request
                    .duration_s
                    .unwrap_or(if cloud { MOCK_AUTO_S } else { MOCK_DEFAULT_S });
            if request.loop_ {
                // A native loop comes back a little long (ElevenLabs: 2.25 s for 2.0 s); a local one is asked for
                // the crossfade's extra seconds, like the real local backend does.
                seconds += if cloud {
                    0.25
                } else {
                    crate::audio::edit::loop_extra_s(seconds)
                };
            }
            let samples = Self::render(
                if cloud {
                    hash(&request.prompt, i)
                } else {
                    seed
                },
                seconds,
                rate,
            );
            if !request.out_dir.as_os_str().is_empty() {
                let _ = std::fs::write(request.out_dir.join(format!("mock-{i}.raw")), b"scratch");
            }
            clips.push(RawClip {
                samples,
                sample_rate: rate,
                channels: 2,
                seed: (!cloud).then_some(seed),
                source_format: if cloud {
                    format!("pcm_{rate}")
                } else {
                    "f32".into()
                },
                credits: cloud.then(|| match request.duration_s {
                    Some(d) => (d * 10.0).ceil() as u32,
                    None => 50,
                }),
                details: serde_json::json!({
                    "mock": true,
                    "batch": { "size": n, "index": i },
                }),
            });
        }
        Ok(clips)
    }
}

/// The two mocks the server registers under the `mock` feature.
pub fn env_backends() -> (MockBackend, MockBackend) {
    let delay = MockBackend::env_delay();
    (
        MockBackend::local(delay).with_env(),
        MockBackend::cloud(delay),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelRegistry;

    fn model(id: &str) -> ModelSpec {
        ModelRegistry::from_toml(MOCK_MODELS)
            .unwrap()
            .get(id)
            .unwrap()
            .clone()
    }

    fn request(variations: u32) -> BackendRequest {
        BackendRequest {
            prompt: "a horn".into(),
            duration_s: Some(1.0),
            variations,
            seed: Some(42),
            sample_rate: 44_100,
            ..Default::default()
        }
    }

    #[test]
    fn mock_models_parse() {
        assert_eq!(model("mock-local").runs(), Runs::Local);
        assert_eq!(model("mock-cloud").runs(), Runs::Cloud);
    }

    #[test]
    fn render_is_deterministic_and_even_seeds_are_tones() {
        let a = MockBackend::render(42, 0.5, 44_100);
        assert_eq!(a, MockBackend::render(42, 0.5, 44_100));
        assert_ne!(a, MockBackend::render(43, 0.5, 44_100));
        assert_eq!(a.len(), 22_050 * 2);
        let peak = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((0.45..=0.5).contains(&peak), "tone peak {peak}");
        // Stereo: both channels equal; fades to silence at the edges.
        assert!(a.chunks(2).all(|f| f[0] == f[1]));
        assert!(a[0].abs() < 0.01 && a[a.len() - 1].abs() < 0.01);
        let noise = MockBackend::render(43, 0.5, 44_100);
        let nmax = noise.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(nmax <= 0.3 && nmax > 0.2, "noise peak {nmax}");
    }

    #[tokio::test]
    async fn local_batch_uses_consecutive_seeds_and_reports_progress() {
        let mock = MockBackend::local(Duration::from_millis(1));
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let progress = ProgressSink::new(move |s| seen2.lock().unwrap().push(s.to_string()));
        let clips = mock
            .generate(
                &model("mock-local"),
                &request(3),
                progress,
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(clips.len(), 3);
        assert_eq!(
            clips.iter().map(|c| c.seed).collect::<Vec<_>>(),
            [Some(42), Some(43), Some(44)]
        );
        assert!(
            clips
                .iter()
                .all(|c| c.sample_rate == 44_100 && c.channels == 2)
        );
        assert!((clips[0].duration_s() - 1.0).abs() < 1e-3);
        assert_eq!(clips[1].details["batch"]["index"], 1);
        assert_eq!(clips[0].credits, None);
        assert_eq!(
            *seen.lock().unwrap(),
            [
                "loading model",
                "generating 1 of 3",
                "generating 2 of 3",
                "generating 3 of 3"
            ]
        );
        assert_eq!(mock.loaded_model().as_deref(), Some("mock-local"));
        assert_eq!(mock.generations.load(Ordering::SeqCst), 1);
        mock.unload().await;
        mock.unload().await;
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 1, "idempotent");
        assert_eq!(mock.loaded_model(), None);
        assert_eq!(mock.provider(), None);
    }

    #[tokio::test]
    async fn unseeded_local_calls_report_a_random_seed() {
        let mock = MockBackend::local(Duration::from_millis(1));
        let mut req = request(1);
        req.seed = None;
        let clips = mock
            .generate(
                &model("mock-local"),
                &req,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(clips[0].seed.is_some_and(|s| s < 1 << 31));
    }

    #[tokio::test]
    async fn cloud_mock_has_no_seeds_and_reports_credits() {
        let mock = MockBackend::cloud(Duration::from_millis(1));
        let clips = mock
            .generate(
                &model("mock-cloud"),
                &request(2),
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(clips.iter().all(|c| c.seed.is_none()));
        assert_eq!(clips[0].credits, Some(10));
        assert_eq!(clips[0].source_format, "pcm_44100");
        assert_ne!(clips[0].samples, clips[1].samples, "variations differ");
        assert_eq!(mock.loaded_model(), None, "a cloud backend holds no model");
        assert_eq!(mock.provider(), Some("mock-cloud"));
        // Automatic length costs 50 credits; a loop comes back 0.25 s long, like ElevenLabs.
        let mut auto = request(1);
        auto.duration_s = None;
        let clip = &mock
            .generate(
                &model("mock-cloud"),
                &auto,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap()[0];
        assert_eq!(clip.credits, Some(50));
        assert!((clip.duration_s() - MOCK_AUTO_S).abs() < 1e-3);
        let mut looped = request(1);
        looped.duration_s = Some(2.0);
        looped.loop_ = true;
        let clip = &mock
            .generate(
                &model("mock-cloud"),
                &looped,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap()[0];
        assert!((clip.duration_s() - 2.25).abs() < 1e-3);
        assert_eq!(clip.credits, Some(20));
    }

    #[tokio::test]
    async fn failures_are_injected_at_a_stage_once() {
        let mock = MockBackend::local(Duration::from_millis(1));
        mock.fail_next("generating", "worker_crashed");
        let err = mock
            .generate(
                &model("mock-local"),
                &request(2),
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        let BackendError::Failed(f) = err else {
            panic!()
        };
        assert_eq!(
            (f.stage.as_str(), f.kind.as_str()),
            ("generating", "worker_crashed")
        );
        assert_eq!(mock.generations.load(Ordering::SeqCst), 0);
        // Only the first job fails.
        assert!(
            mock.generate(
                &model("mock-local"),
                &request(1),
                ProgressSink::noop(),
                CancellationToken::new()
            )
            .await
            .is_ok()
        );
        mock.fail_next("loading", "missing_weights");
        let err = mock
            .generate(
                &model("mock-local"),
                &request(1),
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, BackendError::Failed(f) if f.kind == "missing_weights"));
        assert_eq!(mock.active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancel_stops_generation_and_scratch_is_written_under_out_dir() {
        let scratch = tempfile::tempdir().unwrap();
        let mock = MockBackend::local(Duration::from_millis(20));
        let mut req = request(1);
        req.out_dir = scratch.path().to_path_buf();
        let ok = mock
            .generate(
                &model("mock-local"),
                &req,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(ok.len(), 1);
        assert!(scratch.path().join("mock-0.raw").exists());

        let cancel = CancellationToken::new();
        let c = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            c.cancel();
        });
        let err = mock
            .generate(
                &model("mock-local"),
                &request(4),
                ProgressSink::noop(),
                cancel,
            )
            .await
            .unwrap_err();
        assert_eq!(err, BackendError::Cancelled);
    }

    #[tokio::test]
    async fn availability_can_be_switched() {
        let mock = MockBackend::local(Duration::from_millis(1));
        assert_eq!(
            mock.availability(&model("mock-local")).await,
            Availability::Ready
        );
        mock.set_availability(Availability::Unavailable {
            reason: "no gpu".into(),
        });
        assert_eq!(
            mock.availability(&model("mock-local")).await.reason(),
            Some("no gpu")
        );
    }
}
