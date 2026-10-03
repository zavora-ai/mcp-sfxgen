//! The `generate_sfx` request (design §5) and its validation (R1, R2, R7, R12).
//!
//! `SfxRequest` is what the caller sends (its schema is the tool's input schema, so the field docs are what a model
//! reads). `SfxRequest::resolve` turns it into a `ResolvedRequest`: the model picked, defaults filled in, every
//! parameter checked against that model, the paths planned. Everything wrong with the *call* comes back as a caller
//! error (`isError` result with a fix), never as a protocol error.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};

use crate::audio::{BitDepth, LoopMode, PostOptions};
use crate::backend::BackendRequest;
use crate::config::Config;
use crate::error::SfxError;
use crate::output::{PathPolicy, SfxTarget};
use crate::registry::{BackendOptions, ModelRegistry, ModelSpec};

/// The local models' default length when the caller gives none (design §5).
pub const LOCAL_DEFAULT_DURATION_S: f64 = 2.0;
const MAX_FADE_MS: u32 = 5_000;
/// The longest prompt accepted; far beyond what either model reads.
const MAX_PROMPT_CHARS: usize = 4_000;

/// A request for one or more sound effects.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SfxRequest {
    /// What the sound should be, in plain words: the object, the action, the character ("a cartoon meep, short and
    /// bouncy"). Be concrete about length and tone.
    pub prompt: String,
    /// Model id from `list_models`. Omit for the configured default (a local model unless the config says otherwise).
    /// A cloud model is never picked for you.
    #[serde(default)]
    pub model: Option<String>,
    /// Length in seconds, within the model's range (`list_models` shows it). Omit: local models make 2 s, ElevenLabs
    /// lets the model pick the length (a flat, higher credit price).
    #[serde(default)]
    pub duration_s: Option<f64>,
    /// How many alternatives to make, 1 to 4 (default 1). They are written as `<name>_01`, `<name>_02`, ...
    #[serde(default)]
    #[schemars(range(min = 1, max = 4))]
    pub variations: Option<u32>,
    /// Local models only: the random seed (variation n uses seed + n - 1). Omit for a random one, always reported.
    /// ElevenLabs has no seed: it is ignored there, with a warning.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Make a seamless loop of exactly `duration_s`. Can't be combined with `post.trim_silence` or fades.
    #[serde(default, rename = "loop")]
    pub loop_: bool,
    /// File name stem (letters, digits, `_`, `-`; at most 64). Default: a slug of the prompt.
    #[serde(default)]
    pub name: Option<String>,
    /// Folder for the files, inside an allowed root. Default: the configured output folder.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// An exact `.wav` path inside an allowed root, for one variation only. Existing files are never overwritten.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Cloud models only: 0 to 1, how closely to follow the prompt (the model's default if omitted). Ignored by local
    /// models, with a warning.
    #[serde(default)]
    pub prompt_influence: Option<f64>,
    /// Local models only: sampler steps, 1 to 50 (the model's default if omitted, 8). Ignored by cloud models, with a
    /// warning.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Post-processing of the processed copy (the raw copy is always kept untouched).
    #[serde(default)]
    pub post: Option<PostRequest>,
    /// Wait for the sounds (default true). False returns a job id at once: poll `get_job`, stop it with `cancel_job`.
    #[serde(default = "default_true")]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

/// Post-processing options. The true-peak limiter is on by default; everything else is off.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PostRequest {
    /// Cut leading and trailing silence (below -60 dBFS, 5 ms kept). Not with `loop`.
    #[serde(default)]
    pub trim_silence: Option<bool>,
    /// Fade-in length in milliseconds (0 to 5000). Not with `loop`.
    #[serde(default)]
    pub fade_in_ms: Option<u32>,
    /// Fade-out length in milliseconds (0 to 5000). Not with `loop`.
    #[serde(default)]
    pub fade_out_ms: Option<u32>,
    /// Normalise integrated loudness to this many LUFS (-70 to 0). Clips under 0.4 s have no integrated loudness and
    /// are left alone, with a warning.
    #[serde(default)]
    pub loudness_lufs: Option<f64>,
    /// True-peak ceiling in dBTP (-20 to 0). Omit for the configured default (-1.0); pass null to turn the limiter off.
    #[serde(default, deserialize_with = "double_option")]
    #[schemars(with = "Option<f64>")]
    pub true_peak_dbtp: Option<Option<f64>>,
    /// Mix down to one channel.
    #[serde(default)]
    pub mono: Option<bool>,
    /// Output sample rate: 44100 or 48000. Omit to keep what the model delivers (44100).
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// Output bit depth: 16 (default, dithered) or 24.
    #[serde(default)]
    pub bit_depth: Option<u32>,
}

/// Absent = `None`, `null` = `Some(None)`, a value = `Some(Some(v))`.
fn double_option<'de, D, T>(de: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(de).map(Some)
}

/// What the finisher and the sidecars record about the resolved call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResultParams {
    pub prompt: String,
    pub model: String,
    /// `None`: the model picks (ElevenLabs).
    pub duration_s: Option<f64>,
    pub variations: u32,
    /// The first seed (local models); `None` for cloud models, which have none.
    pub seed: Option<u64>,
    #[serde(rename = "loop")]
    pub loop_: bool,
    pub steps: Option<u32>,
    pub prompt_influence: Option<f64>,
    pub post: PostParams,
}

/// The post-processing options as resolved (defaults filled in).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PostParams {
    pub trim_silence: bool,
    pub fade_in_ms: Option<u32>,
    pub fade_out_ms: Option<u32>,
    pub loudness_lufs: Option<f64>,
    /// `None` = the limiter is off.
    pub true_peak_dbtp: Option<f64>,
    pub mono: bool,
    /// `None` = as delivered.
    pub sample_rate: Option<u32>,
    pub bit_depth: u32,
}

impl From<&PostOptions> for PostParams {
    fn from(o: &PostOptions) -> Self {
        Self {
            trim_silence: o.trim_silence,
            fade_in_ms: o.fade_in_ms,
            fade_out_ms: o.fade_out_ms,
            loudness_lufs: o.loudness_lufs,
            true_peak_dbtp: o.true_peak_dbtp,
            mono: o.mono,
            sample_rate: o.sample_rate,
            bit_depth: u32::from(o.bit_depth.bits()),
        }
    }
}

/// A validated call: everything the job needs, nothing left to decide.
#[derive(Debug, Clone)]
pub struct ResolvedRequest {
    pub model: ModelSpec,
    pub backend: BackendRequest,
    pub post: PostOptions,
    pub loop_mode: LoopMode,
    pub target: SfxTarget,
    pub params: ResultParams,
    /// Things the caller asked for that this model ignores, and the like.
    pub warnings: Vec<String>,
    pub wait: bool,
}

impl SfxRequest {
    /// Check the call against the registry, config and paths. Doesn't look at model status (the server does, with the
    /// backend's availability) and reserves nothing.
    pub fn resolve(
        &self,
        registry: &ModelRegistry,
        cfg: &Config,
        paths: &PathPolicy,
    ) -> Result<ResolvedRequest, SfxError> {
        let prompt = self.prompt.trim();
        if prompt.is_empty() {
            return Err(SfxError::invalid("prompt", "must not be empty"));
        }
        if prompt.chars().count() > MAX_PROMPT_CHARS {
            return Err(SfxError::invalid(
                "prompt",
                format!("is longer than {MAX_PROMPT_CHARS} characters"),
            ));
        }
        let model_id = self
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .unwrap_or(&cfg.default_model);
        let model = registry.get(model_id)?.clone();
        let runs = model.runs();
        let local = runs == crate::backend::Runs::Local;
        let mut warnings = Vec::new();

        let variations = self.variations.unwrap_or(1);
        if !(1..=4).contains(&variations) {
            return Err(SfxError::invalid(
                "variations",
                format!("must be 1 to 4 (got {variations})"),
            ));
        }

        let duration_s = match self.duration_s {
            Some(d) => {
                if !d.is_finite() || !model.duration_s.contains(d) {
                    return Err(SfxError::DurationOutOfRange {
                        model: model.id.clone(),
                        duration_s: d,
                        min: model.duration_s.min,
                        max: model.duration_s.max,
                    });
                }
                Some(d)
            }
            None if local => Some(LOCAL_DEFAULT_DURATION_S),
            None => None,
        };

        if let Some(infl) = self.prompt_influence {
            let in_range = infl.is_finite() && (0.0..=1.0).contains(&infl);
            if !local && !in_range {
                return Err(SfxError::invalid(
                    "prompt_influence",
                    format!("must be between 0 and 1 (got {infl})"),
                ));
            }
            if local {
                warnings.push(format!(
                    "`prompt_influence` is ignored by `{}` (it only applies to cloud models)",
                    model.id
                ));
            }
        }
        let steps = match self.steps {
            Some(steps) if local => {
                if !(1..=50).contains(&steps) {
                    return Err(SfxError::invalid(
                        "steps",
                        format!("must be 1 to 50 (got {steps})"),
                    ));
                }
                Some(steps)
            }
            Some(_) => {
                warnings.push(format!(
                    "`steps` is ignored by `{}` (it only applies to local models)",
                    model.id
                ));
                None
            }
            None => match &model.options {
                BackendOptions::Sa3Mlx { steps, .. } => Some(*steps),
                _ => None,
            },
        };
        let seed = match self.seed {
            Some(seed) if local => {
                if seed >= 1 << 62 {
                    return Err(SfxError::invalid(
                        "seed",
                        "must be below 4611686018427387904 (2^62)",
                    ));
                }
                Some(seed)
            }
            Some(_) => {
                warnings.push(format!(
                    "`seed` is ignored by `{}`: this model has no seed, so results can't be reproduced and the seed is reported as null",
                    model.id
                ));
                None
            }
            None => None,
        };

        let post = self.resolve_post(cfg)?;
        if self.loop_
            && (post.trim_silence
                || post.fade_in_ms.unwrap_or(0) > 0
                || post.fade_out_ms.unwrap_or(0) > 0)
        {
            return Err(SfxError::conflict(
                "`loop` can't be combined with `post.trim_silence` or fades: both would break the seam",
                "drop trim_silence and the fades, or ask for a one-shot sound (loop = false)",
            ));
        }
        let loop_mode = match (self.loop_, model.native_loop) {
            (false, _) => LoopMode::None,
            (true, true) => LoopMode::Native,
            (true, false) => LoopMode::Crossfade {
                // Local models always have a duration (the default fills in).
                target_s: duration_s.unwrap_or(LOCAL_DEFAULT_DURATION_S),
            },
        };

        let target = paths.plan(
            self.output_path.as_deref(),
            self.output_dir.as_deref(),
            self.name.as_deref(),
            prompt,
            variations,
        )?;

        let prompt_influence = (!local).then(|| {
            self.prompt_influence.unwrap_or(match &model.options {
                BackendOptions::Elevenlabs {
                    prompt_influence, ..
                } => *prompt_influence,
                _ => 0.3,
            })
        });
        let backend = BackendRequest {
            prompt: prompt.to_string(),
            duration_s,
            variations,
            seed,
            loop_: self.loop_,
            steps: if local { steps } else { None },
            prompt_influence: if local { None } else { self.prompt_influence },
            sample_rate: post.sample_rate.unwrap_or(44_100),
            out_dir: Default::default(),
        };
        let params = ResultParams {
            prompt: prompt.to_string(),
            model: model.id.clone(),
            duration_s,
            variations,
            seed,
            loop_: self.loop_,
            steps,
            prompt_influence,
            post: PostParams::from(&post),
        };
        Ok(ResolvedRequest {
            model,
            backend,
            post,
            loop_mode,
            target,
            params,
            warnings,
            wait: self.wait,
        })
    }

    fn resolve_post(&self, cfg: &Config) -> Result<PostOptions, SfxError> {
        let p = self.post.clone().unwrap_or_default();
        for (field, ms) in [
            ("post.fade_in_ms", p.fade_in_ms),
            ("post.fade_out_ms", p.fade_out_ms),
        ] {
            if ms.is_some_and(|ms| ms > MAX_FADE_MS) {
                return Err(SfxError::invalid(
                    field,
                    format!("must be 0 to {MAX_FADE_MS}"),
                ));
            }
        }
        if let Some(l) = p.loudness_lufs
            && !(l.is_finite() && (-70.0..=0.0).contains(&l))
        {
            return Err(SfxError::invalid(
                "post.loudness_lufs",
                format!("must be between -70 and 0 (got {l})"),
            ));
        }
        let true_peak_dbtp = match p.true_peak_dbtp {
            None => Some(cfg.post.true_peak_dbtp),
            Some(None) => None,
            Some(Some(v)) => {
                if !(v.is_finite() && (-20.0..=0.0).contains(&v)) {
                    return Err(SfxError::invalid(
                        "post.true_peak_dbtp",
                        format!(
                            "must be between -20 and 0, or null to turn the limiter off (got {v})"
                        ),
                    ));
                }
                Some(v)
            }
        };
        let sample_rate = match p.sample_rate {
            None => None,
            Some(r @ (44_100 | 48_000)) => Some(r),
            Some(other) => {
                return Err(SfxError::invalid(
                    "post.sample_rate",
                    format!("must be 44100 or 48000 (got {other})"),
                ));
            }
        };
        let bit_depth = match p.bit_depth {
            None | Some(16) => BitDepth::Pcm16,
            Some(24) => BitDepth::Pcm24,
            Some(other) => {
                return Err(SfxError::invalid(
                    "post.bit_depth",
                    format!("must be 16 or 24 (got {other})"),
                ));
            }
        };
        Ok(PostOptions {
            trim_silence: p.trim_silence.unwrap_or(false),
            fade_in_ms: p.fade_in_ms.filter(|ms| *ms > 0),
            fade_out_ms: p.fade_out_ms.filter(|ms| *ms > 0),
            loudness_lufs: p.loudness_lufs,
            true_peak_dbtp,
            mono: p.mono.unwrap_or(false),
            sample_rate,
            bit_depth,
        })
    }
}

/// `job_id` argument of `get_job` and `cancel_job`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobIdInput {
    /// The `job_id` returned by `generate_sfx` with `wait: false`.
    pub job_id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MOCK_MODELS;
    use crate::config::MODELS_EXAMPLE;

    struct Fixture {
        _dir: tempfile::TempDir,
        cfg: Config,
        paths: PathPolicy,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().canonicalize().unwrap();
        let cfgdir = base.join(".sfxgen");
        std::fs::create_dir_all(&cfgdir).unwrap();
        std::fs::create_dir_all(base.join("game")).unwrap();
        let cfg = Config::from_toml(
            &cfgdir,
            "allowed_roots = [\"game\"]\ndefault_output_dir = \"game/sfx\"\ndefault_model = \"mock-local\"\n",
        )
        .unwrap();
        let paths = PathPolicy::from_config(&cfg);
        Fixture {
            _dir: dir,
            cfg,
            paths,
        }
    }

    fn registry() -> ModelRegistry {
        ModelRegistry::from_toml(&format!("{MOCK_MODELS}\n{MODELS_EXAMPLE}")).unwrap()
    }

    fn req(json: serde_json::Value) -> SfxRequest {
        serde_json::from_value(json).unwrap()
    }

    fn resolve(f: &Fixture, json: serde_json::Value) -> Result<ResolvedRequest, SfxError> {
        req(json).resolve(&registry(), &f.cfg, &f.paths)
    }

    fn code(r: Result<ResolvedRequest, SfxError>) -> &'static str {
        r.unwrap_err().code()
    }

    #[test]
    fn defaults_for_a_local_model() {
        let f = fixture();
        let r = resolve(&f, serde_json::json!({"prompt": " A cartoon meep! "})).unwrap();
        assert_eq!(r.model.id, "mock-local", "the configured default");
        assert_eq!(r.backend.prompt, "A cartoon meep!");
        assert_eq!(r.backend.duration_s, Some(2.0));
        assert_eq!(r.backend.variations, 1);
        assert_eq!(r.backend.sample_rate, 44_100);
        assert!(r.wait);
        assert_eq!(r.loop_mode, LoopMode::None);
        assert_eq!(r.post, PostOptions::default());
        assert_eq!(r.post.true_peak_dbtp, Some(-1.0));
        assert_eq!(r.target.stem, "a-cartoon-meep");
        assert!(r.target.numbered);
        assert!(r.target.dir.ends_with("game/sfx"));
        assert!(r.warnings.is_empty());
        assert_eq!(r.params.seed, None);
        assert_eq!(r.params.post.bit_depth, 16);
    }

    #[test]
    fn a_cloud_model_picks_its_own_length_and_ignores_local_options_with_warnings() {
        let f = fixture();
        let r = resolve(
            &f,
            serde_json::json!({"prompt": "horn", "model": "elevenlabs-sfx-v2", "seed": 5, "steps": 4}),
        )
        .unwrap();
        assert_eq!(r.backend.duration_s, None);
        assert_eq!(r.backend.seed, None);
        assert_eq!(r.backend.steps, None);
        assert_eq!(r.params.seed, None);
        assert_eq!(r.params.prompt_influence, Some(0.3), "the model's default");
        assert_eq!(r.warnings.len(), 2, "{:?}", r.warnings);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("`seed`") && w.contains("null"))
        );
        assert!(r.warnings.iter().any(|w| w.contains("`steps`")));
        let r = resolve(
            &f,
            serde_json::json!({"prompt": "horn", "model": "elevenlabs-sfx-v2", "prompt_influence": 0.8}),
        )
        .unwrap();
        assert_eq!(r.backend.prompt_influence, Some(0.8));
        assert_eq!(r.params.prompt_influence, Some(0.8));
    }

    #[test]
    fn a_local_model_ignores_prompt_influence_with_a_warning() {
        let f = fixture();
        let r = resolve(
            &f,
            serde_json::json!({"prompt": "horn", "prompt_influence": 0.9, "steps": 4, "seed": 7}),
        )
        .unwrap();
        assert_eq!(r.backend.prompt_influence, None);
        assert_eq!(r.backend.steps, Some(4));
        assert_eq!(r.backend.seed, Some(7));
        assert_eq!(r.params.prompt_influence, None);
        assert!(
            r.warnings[0].contains("`prompt_influence`"),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn the_sa3_default_steps_are_recorded() {
        let f = fixture();
        let r = resolve(
            &f,
            serde_json::json!({"prompt": "horn", "model": "sa3-small-sfx"}),
        )
        .unwrap();
        assert_eq!(r.params.steps, Some(8));
        assert_eq!(r.backend.steps, Some(8));
    }

    #[test]
    fn caller_mistakes_have_their_own_codes() {
        let f = fixture();
        assert_eq!(
            code(resolve(&f, serde_json::json!({"prompt": "  "}))),
            "invalid_params"
        );
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "model": "nope"})
            )),
            "unknown_model"
        );
        for v in [0, 5] {
            assert_eq!(
                code(resolve(
                    &f,
                    serde_json::json!({"prompt": "x", "variations": v})
                )),
                "invalid_params"
            );
        }
        for d in [0.1, 45.0] {
            assert_eq!(
                code(resolve(
                    &f,
                    serde_json::json!({"prompt": "x", "model": "sa3-small-sfx", "duration_s": d})
                )),
                "duration_out_of_range"
            );
        }
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "duration_s": 99.0})
            )),
            "duration_out_of_range",
            "the mock caps at 10 s"
        );
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "model": "elevenlabs-sfx-v2", "prompt_influence": 1.5})
            )),
            "invalid_params"
        );
        assert_eq!(
            code(resolve(&f, serde_json::json!({"prompt": "x", "steps": 0}))),
            "invalid_params"
        );
        assert_eq!(
            code(resolve(&f, serde_json::json!({"prompt": "x", "steps": 51}))),
            "invalid_params"
        );
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "seed": u64::MAX})
            )),
            "invalid_params"
        );
        // Path rules come from the policy.
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "output_path": "game/a.wav", "variations": 2})
            )),
            "option_conflict"
        );
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "output_dir": "/etc"})
            )),
            "path_outside_roots"
        );
        assert_eq!(
            code(resolve(
                &f,
                serde_json::json!({"prompt": "x", "name": "a b"})
            )),
            "invalid_params"
        );
    }

    #[test]
    fn post_options_are_checked_and_resolved() {
        let f = fixture();
        for post in [
            serde_json::json!({"sample_rate": 22050}),
            serde_json::json!({"bit_depth": 32}),
            serde_json::json!({"true_peak_dbtp": 3.0}),
            serde_json::json!({"loudness_lufs": 5.0}),
            serde_json::json!({"fade_in_ms": 60000}),
        ] {
            assert_eq!(
                code(resolve(
                    &f,
                    serde_json::json!({"prompt": "x", "post": post.clone()})
                )),
                "invalid_params",
                "{post}"
            );
        }
        let r = resolve(
            &f,
            serde_json::json!({"prompt": "x", "post": {"trim_silence": true, "fade_in_ms": 10, "fade_out_ms": 0,
                "loudness_lufs": -16.0, "mono": true, "sample_rate": 48000, "bit_depth": 24, "true_peak_dbtp": -2.0}}),
        )
        .unwrap();
        assert_eq!(
            r.post,
            PostOptions {
                trim_silence: true,
                fade_in_ms: Some(10),
                fade_out_ms: None,
                loudness_lufs: Some(-16.0),
                true_peak_dbtp: Some(-2.0),
                mono: true,
                sample_rate: Some(48_000),
                bit_depth: BitDepth::Pcm24,
            }
        );
        assert_eq!(
            r.backend.sample_rate, 48_000,
            "the cloud is asked for 48 kHz up front"
        );
    }

    #[test]
    fn the_limiter_is_a_tri_state() {
        let f = fixture();
        let ceiling = |post: serde_json::Value| {
            resolve(&f, serde_json::json!({"prompt": "x", "post": post}))
                .unwrap()
                .post
                .true_peak_dbtp
        };
        assert_eq!(
            ceiling(serde_json::json!({})),
            Some(-1.0),
            "absent: the configured default"
        );
        assert_eq!(
            ceiling(serde_json::json!({"true_peak_dbtp": null})),
            None,
            "null: off"
        );
        assert_eq!(
            ceiling(serde_json::json!({"true_peak_dbtp": -3.0})),
            Some(-3.0)
        );
        // The config's default is honoured.
        let mut f2 = fixture();
        f2.cfg.post.true_peak_dbtp = -2.5;
        let r = resolve(&f2, serde_json::json!({"prompt": "x"})).unwrap();
        assert_eq!(r.post.true_peak_dbtp, Some(-2.5));
    }

    #[test]
    fn loops_pick_the_right_mode_and_refuse_trim_and_fades() {
        let f = fixture();
        let local = resolve(
            &f,
            serde_json::json!({"prompt": "x", "loop": true, "duration_s": 3.0}),
        )
        .unwrap();
        assert_eq!(local.loop_mode, LoopMode::Crossfade { target_s: 3.0 });
        assert!(local.backend.loop_);
        let default_length = resolve(&f, serde_json::json!({"prompt": "x", "loop": true})).unwrap();
        assert_eq!(
            default_length.loop_mode,
            LoopMode::Crossfade { target_s: 2.0 }
        );
        let cloud = resolve(
            &f,
            serde_json::json!({"prompt": "x", "loop": true, "model": "elevenlabs-sfx-v2"}),
        )
        .unwrap();
        assert_eq!(cloud.loop_mode, LoopMode::Native);
        for post in [
            serde_json::json!({"trim_silence": true}),
            serde_json::json!({"fade_in_ms": 20}),
            serde_json::json!({"fade_out_ms": 20}),
        ] {
            let e = resolve(
                &f,
                serde_json::json!({"prompt": "x", "loop": true, "post": post}),
            )
            .unwrap_err();
            assert_eq!(e.code(), "option_conflict");
            assert!(e.fix().contains("trim_silence"), "{}", e.fix());
        }
        // Zero fades don't count as fades.
        resolve(&f, serde_json::json!({"prompt": "x", "loop": true, "post": {"fade_in_ms": 0, "trim_silence": false}})).unwrap();
    }

    #[test]
    fn unknown_fields_are_rejected_at_parse_time() {
        let err =
            serde_json::from_value::<SfxRequest>(serde_json::json!({"prompt": "x", "duration": 2}));
        assert!(
            err.unwrap_err()
                .to_string()
                .contains("unknown field `duration`")
        );
        let err = serde_json::from_value::<SfxRequest>(
            serde_json::json!({"prompt": "x", "post": {"loud": 1}}),
        );
        assert!(err.is_err());
    }

    #[test]
    fn the_input_schema_documents_the_fields() {
        let schema = serde_json::to_value(schemars::schema_for!(SfxRequest)).unwrap();
        let props = &schema["properties"];
        for field in [
            "prompt",
            "model",
            "duration_s",
            "variations",
            "seed",
            "loop",
            "name",
            "output_dir",
            "output_path",
            "prompt_influence",
            "steps",
            "post",
            "wait",
        ] {
            assert!(
                props[field]["description"]
                    .as_str()
                    .is_some_and(|d| d.len() > 10)
                    || props[field].get("$ref").is_some()
                    || props[field].get("anyOf").is_some(),
                "{field}: {props}"
            );
        }
        assert_eq!(schema["required"], serde_json::json!(["prompt"]));
        assert_eq!(props["variations"]["maximum"], 4);
    }
}
