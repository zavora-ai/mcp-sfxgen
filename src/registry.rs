//! Model registry (design §3): `models.toml` parsing and per-model status (R2, R3, R12).
//!
//! Status never loads anything and never makes a paid call: it combines the backend's availability, whether the licence
//! is accepted in config, whether credentials exist (both passed in), and whether every listed file is in the Hugging
//! Face cache. Precedence: `backend_unavailable`, `needs_licence_acceptance`, `needs_credentials`, `missing_files`,
//! `ready`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backend::Runs;
use crate::error::SfxError;

/// How long a clip may be, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DurationRange {
    pub min: f64,
    pub max: f64,
}

impl DurationRange {
    pub fn contains(&self, seconds: f64) -> bool {
        seconds >= self.min && seconds <= self.max
    }
}

/// Whether commercial use of a component's output is allowed (R3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Commercial {
    Yes,
    No,
    /// Allowed under conditions named in the licence `note` (a revenue threshold, a paid plan, registration).
    Conditional,
}

impl Commercial {
    /// The weakest of the parts: any `no` makes the whole `no`, else any `conditional` makes it `conditional`.
    pub fn combine(parts: impl IntoIterator<Item = Commercial>) -> Commercial {
        parts
            .into_iter()
            .fold(Commercial::Yes, |acc, c| match (acc, c) {
                (Commercial::No, _) | (_, Commercial::No) => Commercial::No,
                (Commercial::Conditional, _) | (_, Commercial::Conditional) => {
                    Commercial::Conditional
                }
                _ => Commercial::Yes,
            })
    }
}

/// One component's licence (R3: every component, and whether commercial use is allowed).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Licence {
    pub component: String,
    pub name: String,
    pub commercial: Commercial,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Files the model needs from one Hugging Face repo.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RepoFiles {
    pub repo: String,
    pub paths: Vec<String>,
    /// Download size of all `paths`, for the hint.
    pub mb: u64,
    /// The repo needs its licence accepted on the Hub (signed in) before it can be downloaded.
    #[serde(default)]
    pub gated: bool,
    /// Pinned revision (commit, tag or branch); default `main`. A pinned revision is looked up exactly: a cache holding
    /// only another revision counts as missing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Per-backend settings, tagged by `kind` (adding a backend adds a variant; adding a model on a backend is config only).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BackendOptions {
    /// Stable Audio 3 on MLX: weight file names (relative to the snapshot's `MLX/` folder) and the default step count.
    Sa3Mlx {
        dit: String,
        decoder: String,
        t5: String,
        #[serde(default = "default_sa3_steps")]
        steps: u32,
    },
    /// ElevenLabs Sound Effects: the API model id and the default prompt influence (0 to 1).
    Elevenlabs {
        model_id: String,
        #[serde(default = "default_prompt_influence")]
        prompt_influence: f64,
    },
    /// The fake backend (tests, demos); the backend id is `mock*`.
    Mock {
        #[serde(default)]
        runs: Runs,
    },
}

fn default_sa3_steps() -> u32 {
    8
}

fn default_prompt_influence() -> f64 {
    0.3
}

impl BackendOptions {
    /// The backend id this variant belongs to.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Sa3Mlx { .. } => "sa3_mlx",
            Self::Elevenlabs { .. } => "elevenlabs",
            Self::Mock { .. } => "mock",
        }
    }

    pub fn runs(&self) -> Runs {
        match self {
            Self::Sa3Mlx { .. } => Runs::Local,
            Self::Elevenlabs { .. } => Runs::Cloud,
            Self::Mock { runs } => *runs,
        }
    }

    /// Local models whose licences must be accepted in config before they run (R12).
    fn needs_config_licence(&self) -> bool {
        matches!(self, Self::Sa3Mlx { .. })
    }

    /// Models that need an API key (R9).
    fn needs_credentials(&self) -> bool {
        matches!(self, Self::Elevenlabs { .. })
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub backend: String,
    pub duration_s: DurationRange,
    #[serde(default)]
    pub native_loop: bool,
    /// Memory a local model needs once loaded, in MB. Unused (0) for cloud models.
    #[serde(default)]
    pub memory_mb: u64,
    /// A typical time for one call, in seconds.
    pub typical_s: f64,
    pub options: BackendOptions,
    #[serde(default)]
    pub files: Vec<RepoFiles>,
    pub licence: Vec<Licence>,
    /// Where the prompt goes, for cloud models (R3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privacy: Option<String>,
}

/// Missing files from one repo, with the command that fetches them.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
pub struct MissingRepo {
    pub repo: String,
    pub files: Vec<String>,
    /// Upper bound: the repo's listed size (exact when every file is missing).
    pub download_mb: u64,
    pub gated: bool,
    /// Where to accept the licence, for gated repos.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accept_at: Option<String>,
    pub command: String,
}

/// A licence the user has to accept before the model runs.
#[derive(Debug, Clone, PartialEq, Serialize, schemars::JsonSchema)]
pub struct LicenceToAccept {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// What `list_models` reports per model (R3).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ModelStatus {
    Ready,
    MissingFiles {
        missing: Vec<MissingRepo>,
        download_mb: u64,
    },
    /// The licences aren't accepted in config yet, or only gated Hub files are missing. Downloads wait for the OK.
    NeedsLicenceAcceptance {
        accept: Vec<LicenceToAccept>,
        /// Where to record the acceptance, e.g. "set `licence_accepted = \"YYYY-MM-DD\"` in the [sa3_mlx] table of config.toml".
        how: String,
        missing: Vec<MissingRepo>,
        download_mb: u64,
    },
    /// No API key in the environment or keychain (only presence is ever reported).
    NeedsCredentials {
        hint: String,
    },
    BackendUnavailable {
        reason: String,
    },
}

impl ModelStatus {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::MissingFiles { .. } => "missing_files",
            Self::NeedsLicenceAcceptance { .. } => "needs_licence_acceptance",
            Self::NeedsCredentials { .. } => "needs_credentials",
            Self::BackendUnavailable { .. } => "backend_unavailable",
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// What to do next, in one message (also the `not ready` error detail).
    pub fn detail(&self) -> String {
        match self {
            Self::Ready => "ready".into(),
            Self::BackendUnavailable { reason } => reason.clone(),
            Self::NeedsCredentials { hint } => hint.clone(),
            Self::MissingFiles {
                missing,
                download_mb,
            } => missing_detail(missing, *download_mb),
            Self::NeedsLicenceAcceptance {
                accept,
                how,
                missing,
                download_mb,
            } => {
                let mut parts = Vec::new();
                if !accept.is_empty() {
                    let names: Vec<String> = accept
                        .iter()
                        .map(|a| match &a.url {
                            Some(url) => format!("{} ({url})", a.name),
                            None => a.name.clone(),
                        })
                        .collect();
                    parts.push(format!("accept {}; then {how}", names.join(" and ")));
                }
                if !missing.is_empty() {
                    parts.push(missing_detail(missing, *download_mb));
                }
                parts.join(". ")
            }
        }
    }

    /// The caller error for a model that can't run: licence and credential statuses get their own codes.
    pub fn not_ready_error(&self, model: &str, alternatives: Vec<String>) -> SfxError {
        match self {
            Self::NeedsLicenceAcceptance { .. } => SfxError::LicenceNotAccepted {
                model: model.to_string(),
                detail: self.detail(),
            },
            Self::NeedsCredentials { .. } => SfxError::CredentialsMissing {
                model: model.to_string(),
                detail: self.detail(),
            },
            _ => SfxError::NotReady {
                model: model.to_string(),
                status: self.code().to_string(),
                detail: self.detail(),
                alternatives,
            },
        }
    }
}

fn missing_detail(missing: &[MissingRepo], download_mb: u64) -> String {
    let steps: Vec<String> = missing
        .iter()
        .map(|m| match &m.accept_at {
            Some(url) => format!(
                "accept the licence at {url}, sign in with `hf auth login`, then run: {}",
                m.command
            ),
            None => format!("run: {}", m.command),
        })
        .collect();
    format!(
        "{} file(s) missing, up to {download_mb} MB to download (needs your OK). {}",
        missing.iter().map(|m| m.files.len()).sum::<usize>(),
        steps.join("; ")
    )
}

/// What a status check needs besides the model itself.
#[derive(Debug, Clone, Copy)]
pub struct StatusContext<'a> {
    pub hf_cache: &'a Path,
    /// The cache `hf download` uses without `--cache-dir`; the command only names `--cache-dir` when they differ.
    pub default_cache: &'a Path,
    /// The backend's reason it can't run here, if any.
    pub backend_unavailable: Option<&'a str>,
    /// `[sa3_mlx] licence_accepted` is set.
    pub licence_accepted: bool,
    /// A key exists in the environment or keychain.
    pub credentials_present: bool,
    /// Where to put the key (`ElevenLabsConfig::credentials_hint`).
    pub credentials_hint: &'a str,
}

impl ModelSpec {
    pub fn runs(&self) -> Runs {
        self.options.runs()
    }

    /// Whether outputs may be used commercially, from the licence of every component (the plan can narrow it later).
    pub fn commercial_use(&self) -> Commercial {
        Commercial::combine(self.licence.iter().map(|l| l.commercial))
    }

    /// Free memory a local model needs before loading: its own size plus the configured headroom (cloud: none).
    pub fn needs_mb_free(&self, headroom_mb: u64) -> Option<u64> {
        (self.runs() == Runs::Local).then(|| self.memory_mb + headroom_mb)
    }

    fn validate(&self) -> Result<(), String> {
        let id = &self.id;
        if id.trim().is_empty() {
            return Err("a model has an empty `id`".into());
        }
        let fail = |msg: String| Err(format!("model `{id}`: {msg}"));
        if self.backend.trim().is_empty() {
            return fail("`backend` is empty".into());
        }
        let kind = self.options.kind();
        let backend_matches = if kind == "mock" {
            self.backend.starts_with("mock")
        } else {
            self.backend == kind
        };
        if !backend_matches {
            return fail(format!(
                "`backend` is `{}` but `options.kind` is `{kind}`",
                self.backend
            ));
        }
        let d = self.duration_s;
        if !(d.min.is_finite() && d.max.is_finite() && d.min > 0.0 && d.max >= d.min) {
            return fail("`duration_s` needs 0 < min <= max".into());
        }
        if !(self.typical_s.is_finite() && self.typical_s >= 0.0) {
            return fail("`typical_s` must be 0 or more".into());
        }
        if self.runs() == Runs::Local && self.memory_mb == 0 {
            return fail("`memory_mb` must be above 0 for a local model".into());
        }
        if self.runs() == Runs::Cloud && self.privacy.as_deref().is_none_or(|p| p.trim().is_empty())
        {
            return fail("a cloud model needs `privacy` saying where the prompt goes".into());
        }
        if self.licence.is_empty() {
            return fail("list the licence of every component in `licence`".into());
        }
        match &self.options {
            BackendOptions::Sa3Mlx {
                dit,
                decoder,
                t5,
                steps,
            } => {
                if *steps == 0 {
                    return fail("options.steps must be at least 1".into());
                }
                for (key, name) in [("dit", dit), ("decoder", decoder), ("t5", t5)] {
                    let listed = self
                        .files
                        .iter()
                        .flat_map(|f| &f.paths)
                        .any(|p| p.rsplit('/').next() == Some(name.as_str()));
                    if !listed {
                        return fail(format!("options.{key} `{name}` is not listed in `files`"));
                    }
                }
            }
            BackendOptions::Elevenlabs {
                model_id,
                prompt_influence,
            } => {
                if model_id.trim().is_empty() {
                    return fail("options.model_id is empty".into());
                }
                if !(0.0..=1.0).contains(prompt_influence) {
                    return fail("options.prompt_influence must be between 0 and 1".into());
                }
            }
            BackendOptions::Mock { .. } => {}
        }
        for files in &self.files {
            let parts: Vec<&str> = files.repo.split('/').collect();
            if parts.len() != 2 || parts.iter().any(|p| p.is_empty()) {
                return fail(format!("repo `{}` must look like `owner/name`", files.repo));
            }
            if files.paths.is_empty() {
                return fail(format!("repo `{}` lists no paths", files.repo));
            }
            for path in &files.paths {
                let p = Path::new(path);
                if path.is_empty()
                    || p.is_absolute()
                    || path.starts_with('/')
                    || path.starts_with('\\')
                    || p.components()
                        .any(|c| matches!(c, std::path::Component::ParentDir))
                {
                    return fail(format!(
                        "`{path}` in `{}` must be a relative path inside the repo",
                        files.repo
                    ));
                }
            }
        }
        Ok(())
    }

    /// Registry status for this model (precedence in the module docs).
    pub fn status(&self, ctx: &StatusContext) -> ModelStatus {
        if let Some(reason) = ctx.backend_unavailable {
            return ModelStatus::BackendUnavailable {
                reason: reason.to_string(),
            };
        }
        let missing = self.missing_files(ctx.hf_cache, ctx.default_cache);
        let download_mb = missing.iter().map(|m| m.download_mb).sum();
        if self.options.needs_config_licence() && !ctx.licence_accepted {
            return ModelStatus::NeedsLicenceAcceptance {
                accept: self
                    .licence
                    .iter()
                    .filter(|l| l.commercial != Commercial::Yes)
                    .map(|l| LicenceToAccept {
                        name: l.name.clone(),
                        url: l.url.clone(),
                    })
                    .collect(),
                how: format!(
                    "set `licence_accepted = \"YYYY-MM-DD\"` in the [{}] table of config.toml",
                    self.options.kind()
                ),
                missing,
                download_mb,
            };
        }
        if self.options.needs_credentials() && !ctx.credentials_present {
            return ModelStatus::NeedsCredentials {
                hint: ctx.credentials_hint.to_string(),
            };
        }
        if missing.is_empty() {
            return ModelStatus::Ready;
        }
        if missing.iter().all(|m| m.gated) {
            ModelStatus::NeedsLicenceAcceptance {
                accept: missing
                    .iter()
                    .map(|m| LicenceToAccept {
                        name: format!("the {} licence on Hugging Face", m.repo),
                        url: m.accept_at.clone(),
                    })
                    .collect(),
                how: "sign in with `hf auth login`".into(),
                missing,
                download_mb,
            }
        } else {
            ModelStatus::MissingFiles {
                missing,
                download_mb,
            }
        }
    }

    /// Files not found in the cache, grouped by repo.
    pub fn missing_files(&self, hf_cache: &Path, default_cache: &Path) -> Vec<MissingRepo> {
        self.files
            .iter()
            .filter_map(|files| {
                let snapshot = snapshot_dir(hf_cache, &files.repo, files.revision.as_deref());
                let absent: Vec<String> = files
                    .paths
                    .iter()
                    .filter(|p| !snapshot.as_ref().is_some_and(|s| s.join(p).is_file()))
                    .cloned()
                    .collect();
                (!absent.is_empty()).then(|| MissingRepo {
                    command: download_command(files, &absent, hf_cache, default_cache),
                    accept_at: files
                        .gated
                        .then(|| format!("https://huggingface.co/{}", files.repo)),
                    repo: files.repo.clone(),
                    files: absent,
                    download_mb: files.mb,
                    gated: files.gated,
                })
            })
            .collect()
    }

    /// The snapshot folder holding this model's files for `files` (the first repo), if it is cached.
    pub fn snapshot(&self, hf_cache: &Path) -> Option<PathBuf> {
        let files = self.files.first()?;
        snapshot_dir(hf_cache, &files.repo, files.revision.as_deref())
    }
}

/// The snapshot an offline load would use.
/// - No revision (or `main`): the commit `refs/main` points at, else the newest snapshot.
/// - A pinned revision: the commit `refs/<revision>` points at, a snapshot named by the revision (a commit), or the
///   one snapshot whose commit starts with it (a short hash). Nothing else: another revision's files don't count.
pub fn snapshot_dir(hf_cache: &Path, repo: &str, revision: Option<&str>) -> Option<PathBuf> {
    let repo_dir = hf_cache.join(format!("models--{}", repo.replace('/', "--")));
    let snapshots = repo_dir.join("snapshots");
    let pinned = revision.filter(|r| *r != "main" && !r.is_empty());
    let via_ref = |name: &str| {
        let commit = std::fs::read_to_string(repo_dir.join("refs").join(name)).ok()?;
        let dir = snapshots.join(commit.trim());
        dir.is_dir().then_some(dir)
    };
    let Some(rev) = pinned else {
        if let Some(dir) = via_ref("main") {
            return Some(dir);
        }
        let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
            .ok()?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let modified = entry.metadata().and_then(|m| m.modified()).ok()?;
                entry.path().is_dir().then(|| (modified, entry.path()))
            })
            .collect();
        dirs.sort_by_key(|d| std::cmp::Reverse(d.0));
        return dirs.into_iter().next().map(|(_, p)| p);
    };
    if let Some(dir) = via_ref(rev) {
        return Some(dir);
    }
    let named = snapshots.join(rev);
    if named.is_dir() {
        return Some(named);
    }
    if rev.len() >= 7 && rev.chars().all(|c| c.is_ascii_hexdigit()) {
        let mut matches = std::fs::read_dir(&snapshots)
            .ok()?
            .filter_map(Result::ok)
            .filter(|e| e.path().is_dir() && e.file_name().to_string_lossy().starts_with(rev))
            .map(|e| e.path());
        let first = matches.next();
        return if matches.next().is_none() {
            first
        } else {
            None
        };
    }
    None
}

/// The exact `hf download` command for the missing files (`--cache-dir` only when the cache isn't the default one).
fn download_command(
    files: &RepoFiles,
    absent: &[String],
    hf_cache: &Path,
    default_cache: &Path,
) -> String {
    let mut parts = vec!["hf".to_string(), "download".into(), quote(&files.repo)];
    parts.extend(absent.iter().map(|p| quote(p)));
    if let Some(rev) = &files.revision {
        parts.push("--revision".into());
        parts.push(quote(rev));
    }
    if hf_cache != default_cache {
        parts.push("--cache-dir".into());
        parts.push(quote(&hf_cache.display().to_string()));
    }
    parts.join(" ")
}

/// Double-quote an argument unless it's plainly safe (double quotes work in sh, PowerShell and cmd).
fn quote(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/:=+@".contains(c))
    {
        arg.to_string()
    } else {
        format!("\"{arg}\"")
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: Vec<ModelSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    #[serde(default)]
    model: Vec<ModelSpec>,
}

/// " (line N)" of the `id = "<id>"` line, or "".
fn locate_model(text: &str, id: &str) -> String {
    let wanted = format!("\"{id}\"");
    text.lines()
        .position(|l| {
            let l = l.trim();
            l.strip_prefix("id")
                .is_some_and(|rest| rest.trim_start().starts_with('=') && l.contains(&wanted))
        })
        .map(|i| format!(" (line {})", i + 1))
        .unwrap_or_default()
}

impl ModelRegistry {
    pub fn load(path: &Path) -> Result<Self, SfxError> {
        let text = std::fs::read_to_string(path).map_err(|e| SfxError::config(path, e))?;
        Self::from_toml(&text).map_err(|msg| SfxError::config(path, msg))
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        let file: RegistryFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut seen = HashSet::new();
        for model in &file.model {
            model
                .validate()
                .map_err(|msg| format!("{msg}{}", locate_model(text, &model.id)))?;
            if !seen.insert(model.id.clone()) {
                return Err(format!(
                    "duplicate model id `{}`{}",
                    model.id,
                    locate_model(text, &model.id)
                ));
            }
        }
        Ok(Self { models: file.model })
    }

    pub fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec, SfxError> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| SfxError::UnknownModel {
                id: id.to_string(),
                available: self.ids(),
            })
    }

    pub fn ids(&self) -> Vec<String> {
        self.models.iter().map(|m| m.id.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;

    const DEFAULT: &str = "/default/hub";
    const REV: &str = "da6edc54ddba10bfd79a077102ded687f80e882b";

    fn registry() -> ModelRegistry {
        ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap()
    }

    fn sa3() -> ModelSpec {
        registry().get("sa3-small-sfx").unwrap().clone()
    }

    fn eleven() -> ModelSpec {
        registry().get("elevenlabs-sfx-v2").unwrap().clone()
    }

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    /// Lay out a repo like huggingface_hub does: `snapshots/<commit>/...` (and `refs/<name>` when given).
    fn cache_repo(cache: &Path, repo: &str, commit: &str, refname: Option<&str>, paths: &[String]) {
        let dir = cache.join(format!("models--{}", repo.replace('/', "--")));
        if let Some(name) = refname {
            std::fs::create_dir_all(dir.join("refs")).unwrap();
            std::fs::write(dir.join("refs").join(name), commit).unwrap();
        }
        for p in paths {
            touch(&dir.join("snapshots").join(commit).join(p));
        }
    }

    fn cache_all(cache: &Path, spec: &ModelSpec) {
        for f in &spec.files {
            cache_repo(
                cache,
                &f.repo,
                f.revision.as_deref().unwrap_or("abc123"),
                None,
                &f.paths,
            );
        }
    }

    fn ctx<'a>(cache: &'a Path, default: &'a Path) -> StatusContext<'a> {
        StatusContext {
            hf_cache: cache,
            default_cache: default,
            backend_unavailable: None,
            licence_accepted: true,
            credentials_present: true,
            credentials_hint: "set ELEVENLABS_API_KEY",
        }
    }

    #[test]
    fn example_registry_parses() {
        let reg = registry();
        assert_eq!(reg.ids(), vec!["sa3-small-sfx", "elevenlabs-sfx-v2"]);
        let m = reg.get("sa3-small-sfx").unwrap();
        assert_eq!(m.backend, "sa3_mlx");
        assert_eq!(m.runs(), Runs::Local);
        assert_eq!(
            m.duration_s,
            DurationRange {
                min: 0.5,
                max: 30.0
            }
        );
        assert!(!m.native_loop);
        assert_eq!((m.memory_mb, m.typical_s), (3700, 1.5));
        assert_eq!(
            m.options,
            BackendOptions::Sa3Mlx {
                dit: "dit_sm-sfx_f16.npz".into(),
                decoder: "same_s_decoder_f32.npz".into(),
                t5: "t5gemma_f16.npz".into(),
                steps: 8
            }
        );
        assert_eq!(m.files.len(), 1);
        assert_eq!(m.files[0].mb, 1626);
        assert_eq!(m.files[0].revision.as_deref(), Some(REV));
        assert_eq!(m.licence.len(), 3);
        assert_eq!(m.commercial_use(), Commercial::Conditional);
        assert_eq!(m.needs_mb_free(1000), Some(4700));
        assert!(m.privacy.is_none());

        let e = reg.get("elevenlabs-sfx-v2").unwrap();
        assert_eq!(e.runs(), Runs::Cloud);
        assert!(e.native_loop);
        assert_eq!(
            e.options,
            BackendOptions::Elevenlabs {
                model_id: "eleven_text_to_sound_v2".into(),
                prompt_influence: 0.3
            }
        );
        assert_eq!(
            e.privacy.as_deref(),
            Some("the prompt is sent to ElevenLabs")
        );
        assert_eq!(e.commercial_use(), Commercial::Conditional);
        assert_eq!(e.needs_mb_free(1000), None);
        assert!(e.files.is_empty());
    }

    #[test]
    fn commercial_combines_to_the_weakest() {
        use Commercial::*;
        assert_eq!(Commercial::combine([Yes, Yes]), Yes);
        assert_eq!(Commercial::combine([Yes, Conditional]), Conditional);
        assert_eq!(Commercial::combine([Conditional, No, Yes]), No);
        assert_eq!(Commercial::combine([]), Yes);
    }

    #[test]
    fn ready_when_everything_is_in_place() {
        let cache = tempfile::tempdir().unwrap();
        let (sa3, eleven) = (sa3(), eleven());
        cache_all(cache.path(), &sa3);
        let c = ctx(cache.path(), Path::new(DEFAULT));
        assert_eq!(sa3.status(&c), ModelStatus::Ready);
        assert_eq!(eleven.status(&c), ModelStatus::Ready);
        assert!(ModelStatus::Ready.is_ready());
    }

    #[test]
    fn pinned_revision_is_found_by_snapshot_name_and_by_refs() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        // The real layout on this Mac: a snapshot named by the commit, no refs/.
        cache_all(cache.path(), &sa3);
        let snap = sa3.snapshot(cache.path()).unwrap();
        assert!(snap.ends_with(REV), "{}", snap.display());
        // Through a ref instead (a tag pointing at the commit).
        let other = tempfile::tempdir().unwrap();
        let f = &sa3.files[0];
        cache_repo(
            other.path(),
            &f.repo,
            "c0ffee0123456789",
            Some(REV),
            &f.paths,
        );
        assert!(
            sa3.snapshot(other.path())
                .unwrap()
                .ends_with("c0ffee0123456789")
        );
    }

    #[test]
    fn a_pinned_revision_ignores_other_snapshots() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        let f = &sa3.files[0];
        cache_repo(
            cache.path(),
            &f.repo,
            "ffffffffffffffffffffffffffffffffffffffff",
            Some("main"),
            &f.paths,
        );
        let status = sa3.status(&ctx(cache.path(), Path::new(DEFAULT)));
        assert_eq!(
            status.code(),
            "missing_files",
            "files of another revision don't count"
        );
    }

    #[test]
    fn short_revision_matches_a_unique_snapshot_prefix() {
        let cache = tempfile::tempdir().unwrap();
        let paths = vec!["a.bin".to_string()];
        cache_repo(cache.path(), "o/r", REV, None, &paths);
        assert!(
            snapshot_dir(cache.path(), "o/r", Some("da6edc54"))
                .unwrap()
                .ends_with(REV)
        );
        assert!(
            snapshot_dir(cache.path(), "o/r", Some("da6edc")).is_none(),
            "too short"
        );
        assert!(snapshot_dir(cache.path(), "o/r", Some("deadbeef")).is_none());
        // Two snapshots sharing the prefix are ambiguous.
        cache_repo(cache.path(), "o/r", "da6edc54ffff", None, &paths);
        assert!(snapshot_dir(cache.path(), "o/r", Some("da6edc54")).is_none());
    }

    #[test]
    fn missing_files_give_sizes_and_exact_commands() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        let f = &sa3.files[0];
        // Only the DiT is cached.
        cache_repo(cache.path(), &f.repo, REV, None, &f.paths[..1]);
        let status = sa3.status(&ctx(cache.path(), cache.path()));
        assert_eq!(status.code(), "missing_files");
        let ModelStatus::MissingFiles {
            missing,
            download_mb,
        } = &status
        else {
            panic!("{status:?}")
        };
        assert_eq!(missing.len(), 1);
        assert_eq!(
            missing[0].command,
            format!(
                "hf download stabilityai/stable-audio-3-optimized MLX/t5gemma_f16.npz MLX/same_s_decoder_f32.npz --revision {REV}"
            )
        );
        assert_eq!(missing[0].files.len(), 2);
        assert_eq!(*download_mb, 1626);
        let detail = status.detail();
        assert!(
            detail.starts_with("2 file(s) missing, up to 1626 MB"),
            "{detail}"
        );
        assert!(detail.contains("needs your OK"), "{detail}");
        // Nothing cached: every file, same command shape; a non-default cache goes into the command.
        let empty = tempfile::tempdir().unwrap();
        let missing = sa3.missing_files(empty.path(), Path::new(DEFAULT));
        assert!(missing[0].command.starts_with(
            "hf download stabilityai/stable-audio-3-optimized MLX/dit_sm-sfx_f16.npz"
        ));
        assert!(
            missing[0].command.contains("--cache-dir"),
            "{}",
            missing[0].command
        );
        assert_eq!(missing[0].files.len(), 3);
    }

    #[test]
    fn licence_not_accepted_comes_before_missing_files() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        let mut c = ctx(cache.path(), Path::new(DEFAULT));
        c.licence_accepted = false;
        let status = sa3.status(&c);
        assert_eq!(status.code(), "needs_licence_acceptance");
        let ModelStatus::NeedsLicenceAcceptance {
            accept,
            how,
            missing,
            ..
        } = &status
        else {
            panic!("{status:?}")
        };
        let names: Vec<&str> = accept.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            names,
            ["Stability AI Community License", "Gemma Terms of Use"]
        );
        assert!(
            how.contains("licence_accepted") && how.contains("[sa3_mlx]"),
            "{how}"
        );
        assert_eq!(missing.len(), 1, "pending downloads are listed too");
        let detail = status.detail();
        assert!(
            detail.contains("https://stability.ai/community-license-agreement"),
            "{detail}"
        );
        assert!(detail.contains("hf download"), "{detail}");
        // With everything cached, only the acceptance is left.
        cache_all(cache.path(), &sa3);
        let status = sa3.status(&c);
        assert_eq!(status.code(), "needs_licence_acceptance");
        assert!(
            !status.detail().contains("hf download"),
            "{}",
            status.detail()
        );
        // ElevenLabs has no config licence gate.
        assert_eq!(eleven().status(&c), ModelStatus::Ready);
    }

    #[test]
    fn only_gated_files_missing_needs_licence_acceptance_on_the_hub() {
        let cache = tempfile::tempdir().unwrap();
        let mut sa3 = sa3();
        sa3.files[0].gated = true;
        let status = sa3.status(&ctx(cache.path(), Path::new(DEFAULT)));
        assert_eq!(status.code(), "needs_licence_acceptance");
        let detail = status.detail();
        assert!(
            detail.contains("https://huggingface.co/stabilityai/stable-audio-3-optimized")
                && detail.contains("hf auth login"),
            "{detail}"
        );
    }

    #[test]
    fn missing_credentials_report_the_hint_only() {
        let cache = tempfile::tempdir().unwrap();
        let mut c = ctx(cache.path(), Path::new(DEFAULT));
        c.credentials_present = false;
        let status = eleven().status(&c);
        assert_eq!(status.code(), "needs_credentials");
        assert_eq!(status.detail(), "set ELEVENLABS_API_KEY");
        // A local model doesn't care.
        cache_all(cache.path(), &sa3());
        assert_eq!(sa3().status(&c), ModelStatus::Ready);
        let err = status.not_ready_error("elevenlabs-sfx-v2", vec![]);
        assert_eq!(err.code(), "credentials_missing");
    }

    #[test]
    fn backend_unavailable_wins() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        let mut c = ctx(cache.path(), Path::new(DEFAULT));
        c.licence_accepted = false;
        c.backend_unavailable = Some("MLX needs Apple Silicon");
        let status = sa3.status(&c);
        assert_eq!(status.code(), "backend_unavailable");
        assert_eq!(status.detail(), "MLX needs Apple Silicon");
        let err = status.not_ready_error("sa3-small-sfx", vec!["elevenlabs-sfx-v2".into()]);
        assert_eq!(err.code(), "model_not_ready");
        let msg = err.to_string();
        assert!(
            msg.contains("(backend_unavailable)")
                && msg.contains("Ready models: elevenlabs-sfx-v2"),
            "{msg}"
        );
    }

    #[test]
    fn not_ready_error_codes_per_status() {
        let cache = tempfile::tempdir().unwrap();
        let sa3 = sa3();
        let mut c = ctx(cache.path(), Path::new(DEFAULT));
        assert_eq!(
            sa3.status(&c).not_ready_error("m", vec![]).code(),
            "model_not_ready"
        );
        c.licence_accepted = false;
        assert_eq!(
            sa3.status(&c).not_ready_error("m", vec![]).code(),
            "licence_not_accepted"
        );
        assert!(
            sa3.status(&c)
                .not_ready_error("m", vec![])
                .is_caller_error()
        );
    }

    #[test]
    fn files_in_a_stale_snapshot_dont_count_without_a_pin() {
        let cache = tempfile::tempdir().unwrap();
        let repo = "o/r";
        let dir = cache.path().join("models--o--r");
        touch(&dir.join("snapshots/old/f.bin"));
        std::fs::create_dir_all(dir.join("snapshots/new")).unwrap();
        std::fs::create_dir_all(dir.join("refs")).unwrap();
        std::fs::write(dir.join("refs/main"), "new\n").unwrap();
        assert_eq!(
            snapshot_dir(cache.path(), repo, None),
            Some(dir.join("snapshots/new"))
        );
        assert_eq!(
            snapshot_dir(cache.path(), repo, Some("main")),
            Some(dir.join("snapshots/new"))
        );
        // No refs: the newest snapshot; a revision naming a snapshot directly: that one.
        std::fs::remove_dir_all(dir.join("refs")).unwrap();
        assert_eq!(
            snapshot_dir(cache.path(), repo, Some("old")),
            Some(dir.join("snapshots/old"))
        );
        assert!(snapshot_dir(cache.path(), repo, None).is_some());
        assert!(snapshot_dir(cache.path(), "x/none", None).is_none());
    }

    #[test]
    fn pinned_revision_goes_into_the_command() {
        let files = RepoFiles {
            repo: "o/r".into(),
            paths: vec!["a b.bin".into()],
            mb: 1,
            gated: false,
            revision: Some("v1.0".into()),
        };
        assert_eq!(
            download_command(&files, &files.paths, Path::new("/c"), Path::new("/c")),
            "hf download o/r \"a b.bin\" --revision v1.0"
        );
    }

    #[test]
    fn validation_names_the_problem_and_the_line() {
        let base = MODELS_EXAMPLE;
        let bad = |from: &str, to: &str| {
            assert!(base.contains(from), "{from}");
            ModelRegistry::from_toml(&base.replacen(from, to, 1)).unwrap_err()
        };
        assert!(bad("memory_mb = 3700", "memory_mb = 0").contains("memory_mb"));
        assert!(bad("min = 0.5, max = 30.0", "min = 5.0, max = 1.0").contains("duration_s"));
        assert!(
            bad(
                "repo = \"stabilityai/stable-audio-3-optimized\"",
                "repo = \"nope\""
            )
            .contains("owner/name")
        );
        assert!(
            bad(
                "\"MLX/dit_sm-sfx_f16.npz\", ",
                "\"../dit_sm-sfx_f16.npz\", "
            )
            .contains("relative path")
        );
        assert!(bad("dit = \"dit_sm-sfx_f16.npz\"", "dit = \"other.npz\"").contains("options.dit"));
        assert!(bad("steps = 8", "steps = 0").contains("steps"));
        assert!(
            bad("prompt_influence = 0.3", "prompt_influence = 1.5").contains("prompt_influence")
        );
        assert!(bad("privacy = \"the prompt is sent to ElevenLabs\"", "").contains("privacy"));
        assert!(bad("backend = \"elevenlabs\"", "backend = \"sa3_mlx\"").contains("options.kind"));
        assert!(bad("id = \"sa3-small-sfx\"", "id = \"\"").contains("empty `id`"));
        let e = bad("memory_mb = 3700", "memory_mb = 0");
        // The id line of the first model in the example file.
        let id_line = MODELS_EXAMPLE
            .lines()
            .position(|l| l.starts_with("id = \"sa3-small-sfx\""))
            .unwrap()
            + 1;
        assert!(
            e.contains("`sa3-small-sfx`") && e.contains(&format!("(line {id_line})")),
            "{e}"
        );
        let dup = format!("{base}\n{}", &base[base.find("[[model]]").unwrap()..]);
        let e = ModelRegistry::from_toml(&dup).unwrap_err();
        assert!(e.contains("duplicate") && e.contains("(line "), "{e}");
    }

    #[test]
    fn typos_and_bad_options_are_parse_errors_with_lines() {
        let typo =
            ModelRegistry::from_toml("[[model]]\nid = \"x\"\nbackend = \"b\"\nqualty = []\n")
                .unwrap_err();
        assert!(typo.contains("qualty") && typo.contains("line 4"), "{typo}");
        let bad_kind = MODELS_EXAMPLE.replace("kind = \"elevenlabs\"", "kind = \"openai\"");
        let e = ModelRegistry::from_toml(&bad_kind).unwrap_err();
        assert!(e.contains("openai") || e.contains("kind"), "{e}");
        let extra_option = MODELS_EXAMPLE.replace(
            "model_id = \"eleven_text_to_sound_v2\"",
            "model_id = \"x\", cfg = 3",
        );
        assert!(ModelRegistry::from_toml(&extra_option).is_err());
    }

    #[test]
    fn mock_models_parse_with_their_lane() {
        let toml = "[[model]]\nid = \"m\"\nbackend = \"mock-cloud\"\nduration_s = { min = 0.5, max = 10 }\ntypical_s = 1\n\
options = { kind = \"mock\", runs = \"cloud\" }\nprivacy = \"nowhere\"\nlicence = [{ component = \"c\", name = \"n\", commercial = \"yes\" }]\n";
        let reg = ModelRegistry::from_toml(toml).unwrap();
        assert_eq!(reg.get("m").unwrap().runs(), Runs::Cloud);
        assert!(ModelRegistry::from_toml(&toml.replace("mock-cloud", "other")).is_err());
    }

    #[test]
    fn unknown_model_lists_available() {
        let err = registry().get("nope").unwrap_err();
        assert_eq!(err.code(), "unknown_model");
        assert!(err.to_string().contains("sa3-small-sfx"), "{err}");
    }

    #[test]
    fn status_serializes_with_a_status_tag() {
        let json =
            serde_json::to_value(ModelStatus::BackendUnavailable { reason: "r".into() }).unwrap();
        assert_eq!(json["status"], "backend_unavailable");
        assert_eq!(
            serde_json::to_value(ModelStatus::Ready).unwrap()["status"],
            "ready"
        );
        let json =
            serde_json::to_value(ModelStatus::NeedsCredentials { hint: "h".into() }).unwrap();
        assert_eq!(json["status"], "needs_credentials");
        let lic = serde_json::to_value(&sa3().licence[0]).unwrap();
        assert_eq!(lic["commercial"], "conditional");
    }

    #[test]
    fn load_reports_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("models.toml");
        std::fs::write(&path, "[[model]]\nid = 1\n").unwrap();
        let err = ModelRegistry::load(&path).unwrap_err();
        assert!(err.to_string().contains("models.toml"), "{err}");
    }
}

/// Run by hand on a machine with the weights: `cargo test real_cache -- --ignored --nocapture`.
#[cfg(test)]
mod live {
    use super::*;
    use crate::config::{MODELS_EXAMPLE, default_hf_cache};

    #[test]
    #[ignore]
    fn real_cache_status() {
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let cache = default_hf_cache();
        for m in reg.models() {
            let status = m.status(&StatusContext {
                hf_cache: &cache,
                default_cache: &cache,
                backend_unavailable: None,
                licence_accepted: true,
                credentials_present: true,
                credentials_hint: "",
            });
            println!("{}: {} ({})", m.id, status.code(), status.detail());
        }
    }
}
