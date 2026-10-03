//! Typed tool outputs (design §5; R1, R3, R4, R6, R10). Every struct derives `JsonSchema`, so each tool advertises the
//! exact shape of its `structuredContent` in `outputSchema` instead of a generic object.

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::backend::{AccountSnapshot, Runs};
use crate::jobs::{JobSnapshot, JobStatus, Lane};
use crate::registry::{Commercial, DurationRange, Licence, LicenceToAccept, MissingRepo};
use crate::request::ResultParams;

/// One finished sound.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FileResult {
    /// The processed WAV.
    pub path: String,
    /// The backend's output, converted to WAV and otherwise untouched (32-bit float for local models, 16-bit PCM as
    /// delivered for ElevenLabs).
    pub raw_path: String,
    /// The provenance JSON: prompt, seed, parameters, measurements, licences, waveform peaks.
    pub sidecar: String,
    pub duration_s: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u32,
    /// Integrated loudness of the processed WAV; null under 0.4 s or for silence.
    pub lufs: Option<f64>,
    pub true_peak_dbtp: f64,
    /// The limiter's largest gain change in dB (0 or negative); null when the limiter was off.
    pub limiter_gain_db: Option<f64>,
    /// For loops: how smooth the wrap is.
    pub seam: Option<SeamResult>,
    /// The seed that made this sound (local models); null for ElevenLabs.
    pub seed: Option<u64>,
}

/// How well a loop wraps.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SeamResult {
    /// The jump at the wrap divided by the signal's typical sample step; about 1 is seamless.
    pub wrap_ratio: f64,
    /// Level of the first 20 ms minus the last 20 ms, in dB.
    pub head_tail_rms_db: f64,
}

impl From<crate::audio::Seam> for SeamResult {
    fn from(s: crate::audio::Seam) -> Self {
        Self {
            wrap_ratio: s.wrap_ratio,
            head_tail_rms_db: s.head_tail_rms_db,
        }
    }
}

/// The result of a finished `generate_sfx` job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GenerateResult {
    pub model: String,
    pub backend: String,
    pub runs: Runs,
    pub files: Vec<FileResult>,
    /// What was resolved: defaults filled in, the post options as applied.
    pub params: ResultParams,
    /// Seconds per phase: queued, gpu_lock_wait, generate, post, write, total.
    pub timings: BTreeMap<String, f64>,
    /// Credits the provider charged for this call (cloud models).
    pub credits: Option<u32>,
    pub commercial_use: Commercial,
    /// Anything the caller should know: ignored options, a limiter that worked hard, skipped steps.
    pub warnings: Vec<String>,
}

/// `generate_sfx` with `wait: false`: the job was queued.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct JobStarted {
    pub job_id: String,
    pub status: JobStatus,
    pub lane: Lane,
    pub progress: Option<String>,
    pub hint: String,
}

/// An `isError` result: what was wrong and what to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ErrorDetail {
    /// Stable code: `invalid_params`, `duration_out_of_range`, `option_conflict`, `path_outside_roots`,
    /// `output_exists`, `model_not_ready`, `licence_not_accepted`, `credentials_missing`, `credit_limit`, `unknown_job`, ...
    pub code: String,
    pub message: String,
    /// The next step.
    pub fix: String,
}

/// What `generate_sfx` can answer with. (The root says `type: object` as well as `anyOf`, because clients expect an
/// object schema for structured content.)
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(untagged)]
#[schemars(extend("type" = "object"))]
// Only ever used to generate the tool's output schema, so the size difference costs nothing.
#[allow(clippy::large_enum_variant)]
pub enum GenerateOutput {
    Done(GenerateResult),
    Started(JobStarted),
    Error(ErrorBody),
}

/// What `get_job` and `cancel_job` can answer with.
#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(untagged)]
#[schemars(extend("type" = "object"))]
pub enum JobOutput {
    Job(JobSnapshot),
    Error(ErrorBody),
}

/// One model in `list_models`.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ModelEntry {
    pub id: String,
    pub name: Option<String>,
    pub backend: String,
    pub runs: Runs,
    /// `ready`, `missing_files`, `needs_licence_acceptance`, `needs_credentials` or `backend_unavailable`.
    pub status: String,
    /// What to do about a status that isn't `ready`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// A local worker holding this model is running now.
    pub loaded: bool,
    pub duration_s: DurationRange,
    pub native_loop: bool,
    /// Typical seconds for one call.
    pub typical_s: f64,
    pub licences: Vec<Licence>,
    /// `yes`, `no` or `conditional`, from the licences and (ElevenLabs) the configured plan.
    pub commercial_use: Commercial,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commercial_note: Option<String>,
    /// Local models: MB the loaded process needs, and the free MB required before loading (with headroom).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs_mb_free: Option<u64>,
    /// Cloud models: where the prompt goes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub privacy: Option<String>,
    /// Cloud models: whether a key was found (never the key).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credentials_present: Option<bool>,
    /// `missing_files` / `needs_licence_acceptance`: what to download, with the exact commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing: Option<Vec<MissingRepo>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accept: Option<Vec<LicenceToAccept>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub how: Option<String>,
    /// `needs_credentials`: where to put the key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
    /// `backend_unavailable`: why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ListModelsOutput {
    pub models: Vec<ModelEntry>,
    pub default_model: String,
    pub available_memory_mb: u64,
    pub allowed_roots: Vec<String>,
    pub default_output_dir: String,
    pub config_dir: String,
    pub hf_cache: String,
    /// Who holds the shared GPU lock right now, if anyone.
    pub gpu_lock_holder: Option<String>,
    pub jobs_busy: usize,
}

/// Credits spent, from the local ledger (it counts every call this server made, whatever the key can read).
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct LedgerTotals {
    pub today: u64,
    pub month: u64,
    pub per_call_limit: u32,
    pub daily_limit: u32,
    pub credits_per_second: u32,
    pub credits_auto_duration: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct AccountStatusOutput {
    pub provider: String,
    /// The plan from config (`free` makes outputs non-commercial); the account's own plan overrides it when readable.
    pub configured_plan: String,
    /// Whether a key was found (never the key).
    pub credentials_present: bool,
    /// False when the key can't read the account (a scoped key without `user_read`) or there is no key.
    pub readable: bool,
    /// Why the account isn't readable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account: Option<AccountSnapshot>,
    pub ledger: LedgerTotals,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct UnloadOutput {
    /// Model ids whose worker was stopped.
    pub unloaded: Vec<String>,
    pub available_memory_mb: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::handler::server::tool::schema_for_output;

    #[test]
    fn every_output_schema_is_an_object_schema() {
        let schemas = [
            ("generate", schema_for_output::<GenerateOutput>()),
            ("job", schema_for_output::<JobOutput>()),
            ("models", schema_for_output::<ListModelsOutput>()),
            ("account", schema_for_output::<AccountStatusOutput>()),
            ("unload", schema_for_output::<UnloadOutput>()),
        ];
        for (name, schema) in schemas {
            assert_eq!(schema["type"], "object", "{name}: {schema:?}");
        }
        // The union lists each shape a caller can see.
        let generate = schema_for_output::<GenerateOutput>();
        assert_eq!(
            generate["anyOf"].as_array().unwrap().len(),
            3,
            "{generate:?}"
        );
        let text = serde_json::to_string(&generate).unwrap();
        for field in [
            "raw_path",
            "limiter_gain_db",
            "commercial_use",
            "job_id",
            "fix",
        ] {
            assert!(
                text.contains(field),
                "{field} missing from the generate schema"
            );
        }
    }

    #[test]
    fn a_result_round_trips() {
        let result = GenerateResult {
            model: "m".into(),
            backend: "b".into(),
            runs: Runs::Local,
            files: vec![],
            params: serde_json::from_value(serde_json::json!({
                "prompt": "p", "model": "m", "duration_s": 1.0, "variations": 1, "seed": null, "loop": false,
                "steps": null, "prompt_influence": null,
                "post": {"trim_silence": false, "fade_in_ms": null, "fade_out_ms": null, "loudness_lufs": null,
                         "true_peak_dbtp": -1.0, "mono": false, "sample_rate": null, "bit_depth": 16}
            }))
            .unwrap(),
            timings: BTreeMap::new(),
            credits: None,
            commercial_use: Commercial::Yes,
            warnings: vec![],
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["params"]["loop"], false);
        assert_eq!(
            serde_json::from_value::<GenerateResult>(json).unwrap(),
            result
        );
    }
}
