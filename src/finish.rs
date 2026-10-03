//! The job finisher (design §6, §7; R1, R7, R10): what happens to the backend's raw clips.
//!
//! Per clip: write the raw copy (`<name>_NN.raw.wav`: 32-bit float for local models, the 16-bit PCM as delivered for
//! ElevenLabs), run the audio pipeline (loop, edits, limiter), write the processed WAV, read it back and measure it
//! (the numbers describe the file on disk), then write the provenance sidecar. Files are written as `.partial` and
//! renamed, names are reserved before the backend runs, and a failure or cancel removes everything this job made.
//! Credits a cloud call spent go into the ledger first thing, so spend is recorded even if writing fails.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::{Value, json};

use crate::audio::{self, BitDepth, Clip, LoopMode, PostOptions};
use crate::backend::{RawClip, Runs, Stage};
use crate::config::Config;
use crate::error::SfxError;
use crate::jobs::{FinishContext, JobFinisher};
use crate::ledger::{Entry as LedgerEntry, Ledger};
use crate::output::{SfxNames, SfxTarget, reserve, write_sidecar};
use crate::registry::{BackendOptions, Commercial, ModelSpec};
use crate::request::{ResolvedRequest, ResultParams};
use crate::result::{FileResult, GenerateResult};

/// Waveform buckets in a sidecar (design §6 step 7).
const PEAK_BUCKETS: usize = 512;

/// Whether outputs of `model` may be used commercially, with the reason when the configured plan narrows it: the
/// ElevenLabs Free plan is non-commercial whatever the model's licence says (R10).
pub fn commercial_for(model: &ModelSpec, cfg: &Config) -> (Commercial, Option<String>) {
    match &model.options {
        BackendOptions::Elevenlabs { .. } if !cfg.elevenlabs.commercial_use() => (
            Commercial::No,
            Some(format!(
                "the configured plan `{}` is non-commercial",
                cfg.elevenlabs.plan
            )),
        ),
        _ => (model.commercial_use(), None),
    }
}

/// Licence notices worth carrying next to the file: every component that isn't a plain yes, with its conditions.
pub fn notices_for(model: &ModelSpec, plan_note: Option<&str>) -> Vec<String> {
    let mut notices: Vec<String> = model
        .licence
        .iter()
        .filter(|l| l.commercial != Commercial::Yes)
        .map(|l| {
            let mut text = format!("{}: {}", l.component, l.name);
            if let Some(note) = &l.note {
                text.push_str(&format!(" ({note})"));
            }
            if let Some(url) = &l.url {
                text.push_str(&format!(", {url}"));
            }
            text
        })
        .collect();
    notices.extend(plan_note.map(str::to_string));
    notices
}

pub struct SfxFinisher {
    model: ModelSpec,
    post: PostOptions,
    loop_mode: LoopMode,
    target: SfxTarget,
    params: ResultParams,
    warnings: Vec<String>,
    commercial: Commercial,
    commercial_note: Option<String>,
    notices: Vec<String>,
    /// The configured plan, recorded for cloud models.
    plan: Option<String>,
    ledger: Ledger,
    names: Mutex<Option<SfxNames>>,
    /// Every final file this job made, so a failure can take them all back.
    written: Arc<Mutex<Vec<PathBuf>>>,
}

impl SfxFinisher {
    pub fn new(resolved: &ResolvedRequest, cfg: &Config, ledger: Ledger) -> Self {
        let (commercial, commercial_note) = commercial_for(&resolved.model, cfg);
        let cloud = resolved.model.runs() == Runs::Cloud;
        Self {
            model: resolved.model.clone(),
            post: resolved.post.clone(),
            loop_mode: resolved.loop_mode,
            target: resolved.target.clone(),
            params: resolved.params.clone(),
            warnings: resolved.warnings.clone(),
            notices: notices_for(&resolved.model, commercial_note.as_deref()),
            commercial,
            commercial_note,
            plan: cloud.then(|| cfg.elevenlabs.plan.clone()),
            ledger,
            names: Mutex::new(None),
            written: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

/// Everything one clip's blocking work needs.
struct ClipWork {
    index: u32,
    raw: RawClip,
    names: SfxNames,
    post: PostOptions,
    loop_mode: LoopMode,
    dither_seed: u64,
    written: Arc<Mutex<Vec<PathBuf>>>,
    progress: crate::backend::ProgressSink,
    sidecar: SidecarBase,
}

/// The parts of a sidecar that don't depend on the clip.
#[derive(Clone)]
struct SidecarBase {
    job_id: String,
    model: ModelSpec,
    backend: String,
    runs: Runs,
    provider: Option<&'static str>,
    params: ResultParams,
    plan: Option<String>,
    commercial: Commercial,
    commercial_note: Option<String>,
    notices: Vec<String>,
    request_warnings: Vec<String>,
    timings: std::collections::BTreeMap<String, f64>,
    batch_size: u32,
}

#[async_trait::async_trait]
impl JobFinisher for SfxFinisher {
    async fn prepare(&self) -> Result<(), SfxError> {
        let target = self.target.clone();
        let variations = self.params.variations;
        let names = tokio::task::spawn_blocking(move || reserve(&target, variations))
            .await
            .map_err(|e| SfxError::Audio {
                message: e.to_string(),
            })??;
        *self.names.lock().unwrap() = Some(names);
        Ok(())
    }

    async fn finish(
        &self,
        clips: Vec<RawClip>,
        ctx: &FinishContext<'_>,
    ) -> Result<Value, SfxError> {
        let started = Instant::now();
        let names = self
            .names
            .lock()
            .unwrap()
            .clone()
            .expect("prepare reserves the names before finish");
        if clips.len() != names.variations as usize {
            return Err(SfxError::Audio {
                message: format!(
                    "the backend returned {} clip(s) for {} variation(s)",
                    clips.len(),
                    names.variations
                ),
            });
        }
        ctx.progress.report(Stage::PostProcessing);

        // Spend first: what the provider charged is gone whether or not the files get written.
        if ctx.runs == Runs::Cloud {
            for (i, clip) in clips.iter().enumerate() {
                if let Some(credits) = clip.credits {
                    let mut entry = LedgerEntry::now(&self.model.id, credits);
                    entry.duration_s = self.params.duration_s;
                    entry.file = Some(names.wav(i as u32 + 1).display().to_string());
                    entry.request_id = clip.details["request_id"].as_str().map(str::to_string);
                    if let Err(e) = self.ledger.append(&entry) {
                        tracing::warn!(error = %crate::secrets::scrub(&e.to_string()), "couldn't record credits in the ledger");
                    }
                }
            }
        }
        let credits: u32 = clips.iter().filter_map(|c| c.credits).sum();
        let credits = (ctx.runs == Runs::Cloud && clips.iter().any(|c| c.credits.is_some()))
            .then_some(credits);

        // The first seed actually used (a random one when the caller gave none), so the result and sidecars can
        // reproduce the call.
        let mut params = self.params.clone();
        if params.seed.is_none() {
            params.seed = clips.first().and_then(|c| c.seed);
        }
        let mut sidecar_timings = ctx.timings.clone();
        sidecar_timings.remove("total_so_far");

        let base = SidecarBase {
            job_id: ctx.job_id.to_string(),
            model: self.model.clone(),
            backend: ctx.backend_id.to_string(),
            runs: ctx.runs,
            provider: ctx.provider,
            params: params.clone(),
            plan: self.plan.clone(),
            commercial: self.commercial,
            commercial_note: self.commercial_note.clone(),
            notices: self.notices.clone(),
            request_warnings: self.warnings.clone(),
            timings: sidecar_timings,
            batch_size: names.variations,
        };
        let mut files = Vec::new();
        let mut warnings = self.warnings.clone();
        for (i, raw) in clips.into_iter().enumerate() {
            if ctx.cancel.is_cancelled() {
                return Err(SfxError::Cancelled);
            }
            let index = i as u32 + 1;
            let work = ClipWork {
                index,
                dither_seed: raw.seed.unwrap_or_else(|| job_hash(ctx.job_id, index)),
                raw,
                names: names.clone(),
                post: self.post.clone(),
                loop_mode: self.loop_mode,
                written: self.written.clone(),
                progress: ctx.progress.clone(),
                sidecar: base.clone(),
            };
            let (file, clip_warnings) = tokio::task::spawn_blocking(move || process_clip(work))
                .await
                .map_err(|e| SfxError::Audio {
                    message: format!("post-processing stopped unexpectedly: {e}"),
                })??;
            let prefix = if names.variations > 1 {
                format!("{}: ", names.item_stem(index))
            } else {
                String::new()
            };
            warnings.extend(clip_warnings.into_iter().map(|w| format!("{prefix}{w}")));
            files.push(file);
        }

        let mut timings = ctx.timings.clone();
        let total_so_far = timings.remove("total_so_far").unwrap_or(0.0);
        let post = started.elapsed().as_secs_f64();
        timings.insert("post".into(), post);
        timings.insert("total".into(), total_so_far + post);
        let result = GenerateResult {
            model: self.model.id.clone(),
            backend: ctx.backend_id.to_string(),
            runs: ctx.runs,
            files,
            params,
            timings,
            credits,
            commercial_use: self.commercial,
            warnings,
        };
        serde_json::to_value(result).map_err(|e| SfxError::Audio {
            message: e.to_string(),
        })
    }

    async fn abort(&self) {
        let names = self.names.lock().unwrap().clone();
        let written: Vec<PathBuf> = std::mem::take(&mut *self.written.lock().unwrap());
        let _ = tokio::task::spawn_blocking(move || {
            for path in written {
                let _ = std::fs::remove_file(path);
            }
            if let Some(names) = names {
                names.remove_partials();
            }
        })
        .await;
    }
}

/// One clip: raw copy, pipeline, processed WAV, measurements, sidecar. Returns the file's result and its warnings.
fn process_clip(work: ClipWork) -> Result<(FileResult, Vec<String>), SfxError> {
    let ClipWork {
        index,
        raw,
        names,
        post,
        loop_mode,
        dither_seed,
        written,
        progress,
        sidecar: base,
    } = work;
    let audio_err = |e: audio::AudioError| match e {
        audio::AudioError::OptionConflict(message) => SfxError::conflict(
            message,
            "drop trim_silence and the fades, or ask for a one-shot sound",
        ),
        other => SfxError::Audio {
            message: other.to_string(),
        },
    };
    let wav = names.wav(index);
    let raw_path = names.raw(index);
    let sidecar_path = names.sidecar(index);
    let note_written = |path: &PathBuf| written.lock().unwrap().push(path.clone());

    let source =
        Clip::new(raw.samples.clone(), raw.sample_rate, raw.channels).map_err(audio_err)?;
    // 1. The raw copy, exactly as delivered.
    let raw_depth = if raw.source_format.starts_with("pcm") {
        BitDepth::Pcm16
    } else {
        BitDepth::Float32
    };
    let raw_partial = names.partial(&raw_path);
    audio::wav::write_wav_exact(&raw_partial, &source, raw_depth).map_err(audio_err)?;
    names.finalize(&raw_path)?;
    note_written(&raw_path);

    // 2. The pipeline.
    let (processed, report) = audio::process(&source, &post, loop_mode).map_err(audio_err)?;

    // 3. The processed WAV (a `.partial` reserved earlier), renamed when complete.
    progress.report(Stage::Writing);
    let partial = names.partial(&wav);
    audio::wav::write_wav(&partial, &processed, post.bit_depth, dither_seed).map_err(audio_err)?;
    names.finalize(&wav)?;
    note_written(&wav);

    // 4. Measure what is on disk.
    let on_disk = audio::wav::read_wav(&wav).map_err(audio_err)?;
    let measured = audio::loudness::measure(&on_disk).map_err(audio_err)?;
    let peaks = audio::peaks::peaks_base64(&on_disk, PEAK_BUCKETS);
    let seam = report.seam.map(Into::into);
    let limiter_gain_db = report.limiter.map(|l| {
        if l.max_gain_reduction_db == 0.0 {
            0.0
        } else {
            -l.max_gain_reduction_db
        }
    });

    let file_name = |p: &PathBuf| {
        p.file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let mut warnings = report.warnings.clone();
    warnings.dedup();
    let mut sidecar_warnings = base.request_warnings.clone();
    sidecar_warnings.extend(warnings.iter().cloned());
    let batch = raw
        .details
        .get("batch")
        .cloned()
        .unwrap_or_else(|| json!({ "size": base.batch_size, "index": index - 1 }));
    let mut source_details = raw.details.clone();
    if let Value::Object(map) = &mut source_details {
        map.remove("batch");
    }
    let doc = json!({
        "generator": format!("mcp-sfxgen {}", env!("CARGO_PKG_VERSION")),
        "created": chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
        "job_id": base.job_id,
        "file": file_name(&wav),
        "raw_file": file_name(&raw_path),
        "model": base.model.id,
        "backend": base.backend,
        "runs": base.runs,
        "provider": base.provider,
        "prompt": base.params.prompt,
        "seed": raw.seed,
        "batch": batch,
        "params": base.params,
        "source": { "format": raw.source_format, "sample_rate": raw.sample_rate, "channels": raw.channels, "details": source_details },
        "post_applied": {
            "steps": report.applied,
            "loudness_gain_db": report.loudness_gain_db,
            "limiter_max_gain_reduction_db": report.limiter.map(|l| l.max_gain_reduction_db),
            "limiter_safety_rescale_db": report.limiter.and_then(|l| l.safety_rescale_db),
        },
        "measurements": {
            "duration_s": on_disk.duration_s(),
            "sample_rate": on_disk.sample_rate,
            "channels": on_disk.channels,
            "bit_depth": post.bit_depth.bits(),
            "lufs": measured.lufs,
            "momentary_max_lufs": measured.momentary_max_lufs,
            "true_peak_dbtp": measured.true_peak_dbtp,
            "sample_peak_dbfs": measured.sample_peak_dbfs,
            "seam": seam,
        },
        "peaks": { "buckets": PEAK_BUCKETS, "format": "int8 min,max pairs per bucket, base64", "data": peaks },
        "timings": base.timings,
        "credits": raw.credits,
        "plan": base.plan,
        "commercial_use": base.commercial,
        "commercial_note": base.commercial_note,
        "licences": base.model.licence,
        "notices": base.notices,
        "warnings": sidecar_warnings,
    });
    write_sidecar(&sidecar_path, &doc)?;
    note_written(&sidecar_path);

    let file = FileResult {
        path: wav.display().to_string(),
        raw_path: raw_path.display().to_string(),
        sidecar: sidecar_path.display().to_string(),
        duration_s: on_disk.duration_s(),
        sample_rate: on_disk.sample_rate,
        channels: on_disk.channels,
        bit_depth: u32::from(post.bit_depth.bits()),
        lufs: measured.lufs,
        true_peak_dbtp: measured.true_peak_dbtp,
        limiter_gain_db,
        seam,
        seed: raw.seed,
    };
    Ok((file, warnings))
}

/// FNV-1a over the job id and clip index: a stable dither seed for backends without seeds.
fn job_hash(job_id: &str, index: u32) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in job_id.bytes().chain(index.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::{MOCK_MODELS, MockBackend};
    use crate::backend::{BackendRequest, ProgressSink, SfxBackend};
    use crate::config::MODELS_EXAMPLE;
    use crate::output::PathPolicy;
    use crate::registry::ModelRegistry;
    use crate::request::SfxRequest;
    use std::collections::BTreeMap;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;

    struct Fixture {
        dir: tempfile::TempDir,
        cfg: Config,
        registry: ModelRegistry,
        ledger: Ledger,
    }

    fn fixture(config_extra: &str) -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let cfgdir = base.join(".sfxgen");
        std::fs::create_dir_all(&cfgdir).unwrap();
        std::fs::create_dir_all(base.join("game")).unwrap();
        let cfg = Config::from_toml(
            &cfgdir,
            &format!(
                "allowed_roots = [\"game\"]\ndefault_output_dir = \"game/sfx\"\ndefault_model = \"mock-local\"\n{config_extra}"
            ),
        )
        .unwrap();
        Fixture {
            ledger: Ledger::open(base.join("credits.jsonl")),
            registry: ModelRegistry::from_toml(&format!("{MOCK_MODELS}\n{MODELS_EXAMPLE}"))
                .unwrap(),
            cfg,
            dir,
        }
    }

    impl Fixture {
        fn resolve(&self, json: serde_json::Value) -> ResolvedRequest {
            let paths = PathPolicy::from_config(&self.cfg);
            serde_json::from_value::<SfxRequest>(json)
                .unwrap()
                .resolve(&self.registry, &self.cfg, &paths)
                .unwrap()
        }

        /// Run the resolved request through a mock backend and the finisher, as the job manager would.
        async fn run(&self, resolved: &ResolvedRequest) -> (SfxFinisher, Result<Value, SfxError>) {
            let finisher = SfxFinisher::new(resolved, &self.cfg, self.ledger.clone());
            finisher.prepare().await.unwrap();
            let backend = if resolved.model.runs() == Runs::Cloud {
                MockBackend::cloud(Duration::from_millis(1))
            } else {
                MockBackend::local(Duration::from_millis(1))
            };
            let mut request: BackendRequest = resolved.backend.clone();
            request.out_dir = self.dir.path().join("scratch");
            std::fs::create_dir_all(&request.out_dir).unwrap();
            let clips = backend
                .generate(
                    &resolved.model,
                    &request,
                    ProgressSink::noop(),
                    CancellationToken::new(),
                )
                .await
                .unwrap();
            let progress = ProgressSink::noop();
            let cancel = CancellationToken::new();
            let ctx = FinishContext {
                job_id: "job_test",
                model: &resolved.model,
                request: &request,
                backend_id: backend.id(),
                runs: backend.runs(),
                provider: backend.provider(),
                timings: BTreeMap::from([
                    ("queued".to_string(), 0.1),
                    ("generate".to_string(), 0.2),
                    ("total_so_far".to_string(), 0.3),
                ]),
                progress: &progress,
                cancel: &cancel,
            };
            let out = finisher.finish(clips, &ctx).await;
            (finisher, out)
        }

        fn files(&self, sub: &str) -> Vec<String> {
            let dir = self
                .dir
                .path()
                .canonicalize()
                .unwrap()
                .join("game")
                .join(sub);
            let mut out: Vec<String> = std::fs::read_dir(dir)
                .map(|d| {
                    d.filter_map(Result::ok)
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect()
                })
                .unwrap_or_default();
            out.sort();
            out
        }
    }

    #[tokio::test]
    async fn a_local_clip_gets_raw_wav_sidecar_and_measurements() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "a cartoon meep", "variations": 2, "seed": 42, "duration_s": 1.0}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert_eq!(result.files.len(), 2);
        assert_eq!(
            f.files("sfx"),
            [
                "a-cartoon-meep_01.json",
                "a-cartoon-meep_01.raw.wav",
                "a-cartoon-meep_01.wav",
                "a-cartoon-meep_02.json",
                "a-cartoon-meep_02.raw.wav",
                "a-cartoon-meep_02.wav"
            ]
        );
        let file = &result.files[0];
        assert_eq!(file.seed, Some(42));
        assert_eq!(result.files[1].seed, Some(43));
        assert_eq!(
            (file.sample_rate, file.channels, file.bit_depth),
            (44_100, 2, 16)
        );
        assert!((file.duration_s - 1.0).abs() < 1e-6);
        assert!(file.true_peak_dbtp <= -0.99, "{}", file.true_peak_dbtp);
        assert_eq!(
            file.limiter_gain_db,
            Some(0.0),
            "a half-scale tone needs no limiting"
        );
        assert!(file.seam.is_none());
        assert_eq!(result.credits, None);
        assert_eq!(result.commercial_use, Commercial::Yes, "the mock's licence");
        assert!(
            result.timings.contains_key("post") && !result.timings.contains_key("total_so_far")
        );
        assert!((result.timings["total"] - (0.3 + result.timings["post"])).abs() < 1e-9);
        assert_eq!(result.timings["queued"], 0.1);

        // The raw copy is 32-bit float and the processed one 16-bit.
        let raw = hound::WavReader::open(&file.raw_path).unwrap().spec();
        assert_eq!(
            (raw.bits_per_sample, raw.sample_format),
            (32, hound::SampleFormat::Float)
        );
        let wav = hound::WavReader::open(&file.path).unwrap().spec();
        assert_eq!(
            (wav.bits_per_sample, wav.sample_format),
            (16, hound::SampleFormat::Int)
        );

        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&file.sidecar).unwrap()).unwrap();
        assert_eq!(
            sidecar["generator"],
            format!("mcp-sfxgen {}", env!("CARGO_PKG_VERSION"))
        );
        assert_eq!(sidecar["file"], "a-cartoon-meep_01.wav");
        assert_eq!(sidecar["raw_file"], "a-cartoon-meep_01.raw.wav");
        assert_eq!(sidecar["model"], "mock-local");
        assert_eq!(
            (sidecar["backend"].as_str(), sidecar["runs"].as_str()),
            (Some("mock"), Some("local"))
        );
        assert_eq!(sidecar["prompt"], "a cartoon meep");
        assert_eq!(sidecar["seed"], 42);
        assert_eq!(sidecar["batch"], serde_json::json!({"size": 2, "index": 0}));
        assert!(result.files[1].path.ends_with("_02.wav"));
        assert_eq!(sidecar["params"]["variations"], 2);
        assert_eq!(
            sidecar["post_applied"]["steps"],
            serde_json::json!(["limiter:-1.0dBTP"])
        );
        assert_eq!(sidecar["measurements"]["sample_rate"], 44_100);
        assert!(sidecar["measurements"]["lufs"].as_f64().unwrap() < 0.0);
        assert_eq!(sidecar["peaks"]["buckets"], 512);
        let peaks = sidecar["peaks"]["data"].as_str().unwrap();
        assert_eq!(base64_len(peaks), 1024);
        assert_eq!(sidecar["credits"], Value::Null);
        assert_eq!(sidecar["plan"], Value::Null);
        assert_eq!(sidecar["commercial_use"], "yes");
        assert_eq!(sidecar["licences"][0]["name"], "MIT");
        assert!(sidecar["warnings"].as_array().unwrap().is_empty());
        assert_eq!(sidecar["timings"]["queued"], 0.1);
    }

    fn base64_len(s: &str) -> usize {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .unwrap()
            .len()
    }

    #[tokio::test]
    async fn a_random_seed_is_reported_in_the_params_and_sidecars() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "duration_s": 1.0}));
        assert_eq!(resolved.params.seed, None);
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        let seed = result.files[0].seed.expect("a local clip has a seed");
        assert_eq!(result.params.seed, Some(seed));
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&result.files[0].sidecar).unwrap()).unwrap();
        assert_eq!(sidecar["params"]["seed"], seed);
        assert!(sidecar["timings"].get("total_so_far").is_none());
    }

    #[tokio::test]
    async fn sa3_outputs_are_conditionally_commercial_with_the_licence_notes() {
        let f = fixture("");
        let resolved = f.resolve(
            serde_json::json!({"prompt": "meep", "model": "sa3-small-sfx", "duration_s": 1.0}),
        );
        // The mock backend stands in for the model; what matters here is the metadata.
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert_eq!(result.commercial_use, Commercial::Conditional);
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&result.files[0].sidecar).unwrap()).unwrap();
        assert_eq!(sidecar["commercial_use"], "conditional");
        let notices: Vec<&str> = sidecar["notices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect();
        assert!(
            notices
                .iter()
                .any(|n| n.contains("Stability AI Community License") && n.contains("US$1M")),
            "{notices:?}"
        );
        assert!(notices.iter().any(|n| n.contains("Gemma")), "{notices:?}");
        assert_eq!(sidecar["licences"].as_array().unwrap().len(), 3);
        assert_eq!(sidecar["params"]["steps"], 8);
    }

    #[tokio::test]
    async fn a_cloud_clip_records_credits_plan_and_a_non_commercial_free_plan() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "horn", "model": "mock-cloud", "duration_s": 1.0, "variations": 2}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert_eq!(result.credits, Some(20));
        assert_eq!(f.ledger.today_total().unwrap(), 20);
        let ledger = std::fs::read_to_string(f.ledger.path()).unwrap();
        assert_eq!(ledger.lines().count(), 2);
        assert!(ledger.contains("horn_01.wav"), "{ledger}");
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&result.files[0].sidecar).unwrap()).unwrap();
        assert_eq!(sidecar["credits"], 10);
        assert_eq!(sidecar["plan"], "free");
        assert_eq!(sidecar["provider"], "mock-cloud");
        assert_eq!(sidecar["seed"], Value::Null);
        // Source audio from a PCM backend is kept as 16-bit in the raw copy.
        let raw = hound::WavReader::open(&result.files[0].raw_path)
            .unwrap()
            .spec();
        assert_eq!(
            (raw.bits_per_sample, raw.sample_format),
            (16, hound::SampleFormat::Int)
        );
    }

    #[tokio::test]
    async fn elevenlabs_on_the_free_plan_is_not_commercial_and_a_paid_plan_is() {
        let f = fixture("");
        let model = f.registry.get("elevenlabs-sfx-v2").unwrap();
        let (c, note) = commercial_for(model, &f.cfg);
        assert_eq!(c, Commercial::No);
        assert!(note.unwrap().contains("`free`"));
        let paid = fixture("[elevenlabs]\nplan = \"creator\"\n");
        let (c, note) = commercial_for(paid.registry.get("elevenlabs-sfx-v2").unwrap(), &paid.cfg);
        assert_eq!(
            c,
            Commercial::Conditional,
            "paid plans: the model's own licence applies"
        );
        assert!(note.is_none());
        assert_eq!(
            commercial_for(f.registry.get("sa3-small-sfx").unwrap(), &f.cfg).0,
            Commercial::Conditional
        );
    }

    #[tokio::test]
    async fn the_raw_copy_of_pcm_audio_is_bit_exact() {
        let f = fixture("");
        let resolved = f.resolve(
            serde_json::json!({"prompt": "horn", "model": "mock-cloud", "duration_s": 1.0}),
        );
        // The mock cloud renders floats; quantise them the way a PCM backend delivers, then compare after the round trip.
        let finisher = SfxFinisher::new(&resolved, &f.cfg, f.ledger.clone());
        finisher.prepare().await.unwrap();
        let pcm: Vec<u8> = (0..8000u32)
            .flat_map(|i| (((i * 91) % 65536) as u16).to_le_bytes())
            .collect();
        let source = Clip::from_s16le_bytes(&pcm, 44_100, 2).unwrap();
        let clip = RawClip {
            samples: source.samples.clone(),
            sample_rate: 44_100,
            channels: 2,
            seed: None,
            source_format: "pcm_44100".into(),
            credits: Some(10),
            details: json!({"request_id": "r1", "batch": {"size": 1, "index": 0}}),
        };
        let progress = ProgressSink::noop();
        let cancel = CancellationToken::new();
        let request = resolved.backend.clone();
        let ctx = FinishContext {
            job_id: "j",
            model: &resolved.model,
            request: &request,
            backend_id: "mock-cloud",
            runs: Runs::Cloud,
            provider: Some("mock-cloud"),
            timings: BTreeMap::new(),
            progress: &progress,
            cancel: &cancel,
        };
        let out: GenerateResult =
            serde_json::from_value(finisher.finish(vec![clip], &ctx).await.unwrap()).unwrap();
        let back = audio::wav::read_wav(std::path::Path::new(&out.files[0].raw_path)).unwrap();
        assert_eq!(
            back.samples, source.samples,
            "bit for bit, no dither on the raw copy"
        );
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&out.files[0].sidecar).unwrap()).unwrap();
        assert_eq!(sidecar["source"]["format"], "pcm_44100");
        assert_eq!(sidecar["source"]["details"]["request_id"], "r1");
    }

    #[tokio::test]
    async fn loops_report_the_seam_and_have_the_exact_length() {
        let f = fixture("");
        let resolved = f.resolve(
            serde_json::json!({"prompt": "engine hum", "loop": true, "duration_s": 2.0, "seed": 2}),
        );
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        let file = &result.files[0];
        assert!((file.duration_s - 2.0).abs() < 1e-6, "{}", file.duration_s);
        assert!(file.seam.is_some());
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&file.sidecar).unwrap()).unwrap();
        assert!(
            sidecar["post_applied"]["steps"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s == "loop_crossfade")
        );
        assert!(sidecar["measurements"]["seam"]["wrap_ratio"].is_number());
        // A native loop (cloud) is only measured.
        let resolved = f.resolve(serde_json::json!({"prompt": "engine hum", "loop": true, "duration_s": 2.0, "model": "mock-cloud", "name": "hum-cloud"}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert!(
            (result.files[0].duration_s - 2.25).abs() < 1e-6,
            "kept as delivered"
        );
        assert!(result.files[0].seam.is_some());
    }

    #[tokio::test]
    async fn post_options_are_applied_and_reported() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "duration_s": 1.0, "seed": 4,
            "post": {"mono": true, "sample_rate": 48000, "bit_depth": 24, "loudness_lufs": -14.0, "fade_out_ms": 20}}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        let file = &result.files[0];
        assert_eq!(
            (file.channels, file.sample_rate, file.bit_depth),
            (1, 48_000, 24)
        );
        let spec = hound::WavReader::open(&file.path).unwrap().spec();
        assert_eq!(spec.bits_per_sample, 24);
        assert!(file.true_peak_dbtp <= -0.99);
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&file.sidecar).unwrap()).unwrap();
        let steps = sidecar["post_applied"]["steps"].to_string();
        assert!(
            steps.contains("mono")
                && steps.contains("resample:48000")
                && steps.contains("loudness:-14.0LUFS"),
            "{steps}"
        );
    }

    #[tokio::test]
    async fn a_hot_clip_is_limited_and_the_gain_is_reported() {
        let f = fixture("");
        // A loudness target that pushes the peaks past the ceiling makes the limiter work.
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "duration_s": 1.0, "seed": 4, "post": {"loudness_lufs": 0.0}}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        let file = &result.files[0];
        assert!(file.true_peak_dbtp <= -0.99, "{}", file.true_peak_dbtp);
        assert!(
            file.limiter_gain_db.unwrap() < -1.0,
            "{:?}",
            file.limiter_gain_db
        );
        // Turning the limiter off reports no gain.
        let resolved = f.resolve(serde_json::json!({"prompt": "meep two", "duration_s": 1.0, "seed": 4, "post": {"true_peak_dbtp": null}}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert_eq!(result.files[0].limiter_gain_db, None);
    }

    #[tokio::test]
    async fn request_warnings_reach_the_result_and_the_sidecar() {
        let f = fixture("");
        let resolved = f.resolve(
            serde_json::json!({"prompt": "horn", "model": "mock-cloud", "duration_s": 1.0}),
        );
        assert!(
            resolved.warnings.is_empty(),
            "the mock cloud takes seeds and steps silently only if asked"
        );
        let mut resolved = resolved;
        resolved.warnings.push("`seed` is ignored".into());
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert_eq!(result.warnings, ["`seed` is ignored"]);
        let sidecar: Value =
            serde_json::from_slice(&std::fs::read(&result.files[0].sidecar).unwrap()).unwrap();
        assert_eq!(sidecar["warnings"][0], "`seed` is ignored");
    }

    #[tokio::test]
    async fn abort_removes_everything_this_job_made() {
        let f = fixture("");
        let resolved =
            f.resolve(serde_json::json!({"prompt": "meep", "variations": 2, "duration_s": 1.0}));
        let (fin, out) = f.run(&resolved).await;
        out.unwrap();
        assert_eq!(f.files("sfx").len(), 6);
        fin.abort().await;
        assert!(f.files("sfx").is_empty(), "{:?}", f.files("sfx"));
        fin.abort().await; // idempotent

        // After prepare only (a failed or cancelled backend): the reservation markers go.
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "variations": 2}));
        let fin = SfxFinisher::new(&resolved, &f.cfg, f.ledger.clone());
        fin.prepare().await.unwrap();
        assert_eq!(
            f.files("sfx"),
            ["meep_01.wav.partial", "meep_02.wav.partial"]
        );
        fin.abort().await;
        assert!(f.files("sfx").is_empty());
    }

    #[tokio::test]
    async fn a_wrong_number_of_clips_is_an_error_not_a_partial_result() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "variations": 2}));
        let fin = SfxFinisher::new(&resolved, &f.cfg, f.ledger.clone());
        fin.prepare().await.unwrap();
        let progress = ProgressSink::noop();
        let cancel = CancellationToken::new();
        let request = resolved.backend.clone();
        let ctx = FinishContext {
            job_id: "j",
            model: &resolved.model,
            request: &request,
            backend_id: "mock",
            runs: Runs::Local,
            provider: None,
            timings: BTreeMap::new(),
            progress: &progress,
            cancel: &cancel,
        };
        let err = fin.finish(vec![], &ctx).await.unwrap_err();
        assert_eq!(err.code(), "audio_error");
        fin.abort().await;
        assert!(f.files("sfx").is_empty());
    }

    #[tokio::test]
    async fn an_exact_output_path_is_used_as_given() {
        let f = fixture("");
        let resolved = f.resolve(serde_json::json!({"prompt": "meep", "output_path": "game/audio/meep.wav", "duration_s": 1.0}));
        let (_fin, out) = f.run(&resolved).await;
        let result: GenerateResult = serde_json::from_value(out.unwrap()).unwrap();
        assert!(result.files[0].path.ends_with("game/audio/meep.wav"));
        assert_eq!(f.files("audio"), ["meep.json", "meep.raw.wav", "meep.wav"]);
        // A second call to the same exact path is refused when the job starts.
        let again = SfxFinisher::new(&resolved, &f.cfg, f.ledger.clone());
        assert_eq!(again.prepare().await.unwrap_err().code(), "output_exists");
    }
}
