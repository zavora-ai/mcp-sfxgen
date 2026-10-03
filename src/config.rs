//! Server configuration (design §3): config dir discovery, allowed roots, limits, and the `[sa3_mlx]`, `[elevenlabs]`
//! and `[post]` tables (R2, R10, R12). Errors carry line numbers: TOML syntax and unknown keys through the parser,
//! value checks through `locate`.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::SfxError;

pub const CONFIG_EXAMPLE: &str = include_str!("../config.example.toml");
pub const MODELS_EXAMPLE: &str = include_str!("../models.example.toml");

/// Environment variable overriding the config directory.
pub const CONFIG_DIR_ENV: &str = "MCP_SFXGEN_CONFIG_DIR";
/// Environment variable overriding the cache root (tests and sandboxes). It moves the shared GPU lock too, so a server
/// started with it no longer coordinates with sister servers.
pub const CACHE_DIR_ENV: &str = "MCP_SFXGEN_CACHE_DIR";
/// Environment variable overriding the data root (the credit ledger lives under it).
pub const DATA_DIR_ENV: &str = "MCP_SFXGEN_DATA_DIR";

const APP_DIR: &str = "mcp-sfxgen";
const DEFAULT_ROOT: &str = "~/Documents/mcp-sfxgen";
const DEFAULT_MODEL: &str = "sa3-small-sfx";
const DEFAULT_HEAVY: [&str; 6] = [
    "UnrealEditor",
    "Blender",
    "VoiceStudio",
    "sd-server",
    "trellis",
    "mcp-meshgen",
];

#[derive(Debug, Clone)]
pub struct Config {
    pub config_dir: PathBuf,
    /// The platform cache root (`~/Library/Caches`, `$XDG_CACHE_HOME`, `%LOCALAPPDATA%`), shared with sister servers.
    pub cache_root: PathBuf,
    /// The platform data root (`~/Library/Application Support`, `$XDG_DATA_HOME`, `%APPDATA%`).
    pub data_root: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
    /// Always inside an allowed root.
    pub default_output_dir: PathBuf,
    pub default_model: String,
    pub hf_cache: PathBuf,
    pub memory_headroom_mb: u64,
    pub heavy_processes: Vec<String>,
    pub job_ttl_s: u64,
    /// Jobs waiting per lane before `queue_full`.
    pub queue_depth: usize,
    /// 0 = unload only on `unload_models`.
    pub idle_unload_s: u64,
    pub sa3_mlx: Sa3MlxConfig,
    pub elevenlabs: ElevenLabsConfig,
    pub post: PostConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sa3MlxConfig {
    /// The `optimized/mlx` folder of the stable-audio-3 clone.
    pub dir: PathBuf,
    /// The date the licences were accepted; `None` = not accepted.
    pub licence_accepted: Option<chrono::NaiveDate>,
}

impl Sa3MlxConfig {
    /// `<dir>/.venv/bin/python` (`<dir>\.venv\Scripts\python.exe` on Windows).
    pub fn python_path(&self) -> PathBuf {
        venv_python(&self.dir)
    }

    pub fn licence_is_accepted(&self) -> bool {
        self.licence_accepted.is_some()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ElevenLabsConfig {
    pub api_key_env: String,
    pub keychain_service: String,
    /// Lower-cased plan name; only `free` is known to be non-commercial.
    pub plan: String,
    pub concurrency: usize,
    pub per_call_credit_limit: u32,
    pub daily_credit_limit: u32,
    pub credits_per_second: u32,
    pub credits_auto_duration: u32,
    pub timeout_s: u64,
}

impl ElevenLabsConfig {
    /// ElevenLabs' terms: the Free plan is non-commercial (R10).
    pub fn commercial_use(&self) -> bool {
        self.plan != "free"
    }

    /// Where to put the key, for `needs_credentials` and `credentials_missing` messages.
    pub fn credentials_hint(&self) -> String {
        format!(
            "set the {} environment variable, or store the key in the {} item ({}); the key is never written to config",
            self.api_key_env,
            self.keychain_service,
            if cfg!(target_os = "macos") {
                "macOS Keychain: `security add-generic-password -s <service> -a <account> -w`"
            } else {
                "the OS keyring"
            }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostConfig {
    pub true_peak_dbtp: f64,
}

/// The interpreter inside `<dir>/.venv` for this platform.
pub fn venv_python(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        dir.join(".venv").join("bin").join("python")
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    allowed_roots: Option<Vec<String>>,
    default_output_dir: Option<String>,
    default_model: Option<String>,
    hf_cache: Option<String>,
    memory_headroom_mb: Option<u64>,
    heavy_processes: Option<Vec<String>>,
    job_ttl_s: Option<u64>,
    queue_depth: Option<usize>,
    idle_unload_s: Option<u64>,
    #[serde(default)]
    sa3_mlx: RawSa3Mlx,
    #[serde(default)]
    elevenlabs: RawElevenLabs,
    #[serde(default)]
    post: RawPost,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSa3Mlx {
    dir: Option<String>,
    licence_accepted: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawElevenLabs {
    api_key_env: Option<String>,
    keychain_service: Option<String>,
    plan: Option<String>,
    concurrency: Option<usize>,
    per_call_credit_limit: Option<u32>,
    daily_credit_limit: Option<u32>,
    credits_per_second: Option<u32>,
    credits_auto_duration: Option<u32>,
    timeout_s: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPost {
    true_peak_dbtp: Option<f64>,
}

/// " (line N)" for the first `key = ...` line in `table` (`None` = before any table header), or "".
pub fn locate(text: &str, table: Option<&str>, key: &str) -> String {
    let mut current: Option<&str> = None;
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if let Some(header) = line.strip_prefix('[') {
            current = Some(header.trim_end_matches(']').trim());
            continue;
        }
        let in_table = current == table;
        let is_key = line
            .strip_prefix(key)
            .is_some_and(|rest| rest.trim_start().starts_with('='));
        if in_table && is_key {
            return format!(" (line {})", i + 1);
        }
    }
    String::new()
}

impl Config {
    /// Resolve the config directory: `$MCP_SFXGEN_CONFIG_DIR`, else `<platform config dir>/mcp-sfxgen`.
    pub fn default_dir() -> PathBuf {
        let cwd = std::env::current_dir().unwrap_or_else(|_| home_dir());
        config_dir_from(non_empty_env(CONFIG_DIR_ENV), dirs::config_dir(), &cwd)
    }

    /// Load `config.toml` from `dir`, writing the example config and model registry first if absent.
    pub fn load_or_init(dir: &Path) -> Result<Self, SfxError> {
        std::fs::create_dir_all(dir).map_err(|e| SfxError::config(dir, e))?;
        let config_path = dir.join("config.toml");
        if !config_path.exists() {
            std::fs::write(&config_path, CONFIG_EXAMPLE)
                .map_err(|e| SfxError::config(&config_path, e))?;
        }
        let models_path = dir.join("models.toml");
        if !models_path.exists() {
            std::fs::write(&models_path, MODELS_EXAMPLE)
                .map_err(|e| SfxError::config(&models_path, e))?;
        }
        let text =
            std::fs::read_to_string(&config_path).map_err(|e| SfxError::config(&config_path, e))?;
        Self::from_toml(dir, &text).map_err(|msg| SfxError::config(&config_path, msg))
    }

    /// Parse config text; `dir` is recorded as the config directory and anchors relative paths.
    pub fn from_toml(dir: &Path, text: &str) -> Result<Self, String> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| e.to_string())?;
        let base = dir.parent().unwrap_or(dir);
        let resolve = |p: &str| resolve_path(base, p);
        let non_empty = |v: Option<String>| v.filter(|s| !s.trim().is_empty());
        let bad = |table: Option<&str>, key: &str, message: String| {
            let name = table.map_or(key.to_string(), |t| format!("{t}.{key}"));
            format!("{name} {message}{}", locate(text, table, key))
        };

        let allowed_roots: Vec<PathBuf> = match raw.allowed_roots {
            Some(roots) if roots.iter().any(|r| !r.trim().is_empty()) => roots
                .iter()
                .filter(|r| !r.trim().is_empty())
                .map(|r| resolve(r))
                .collect(),
            _ => vec![expand_tilde(DEFAULT_ROOT)],
        };
        let default_output_dir = match non_empty(raw.default_output_dir) {
            Some(d) => {
                let out = resolve(&d);
                if !allowed_roots.iter().any(|root| out.starts_with(root)) {
                    return Err(bad(
                        None,
                        "default_output_dir",
                        format!("{} is not inside any allowed_roots entry", out.display()),
                    ));
                }
                out
            }
            None => allowed_roots[0].join("sfx"),
        };

        let queue_depth = raw.queue_depth.unwrap_or(4);
        if queue_depth == 0 {
            return Err(bad(None, "queue_depth", "must be at least 1".into()));
        }

        let sa3 = raw.sa3_mlx;
        let licence_accepted = match non_empty(sa3.licence_accepted) {
            None => None,
            Some(s) => Some(
                chrono::NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").map_err(|_| {
                    bad(
                        Some("sa3_mlx"),
                        "licence_accepted",
                        format!("must be empty or a date like 2026-10-03 (got `{s}`)"),
                    )
                })?,
            ),
        };
        let sa3_mlx = Sa3MlxConfig {
            dir: resolve(
                non_empty(sa3.dir)
                    .as_deref()
                    .unwrap_or("~/Developer/unreal_engine/tools/stable-audio-3/optimized/mlx"),
            ),
            licence_accepted,
        };

        let el = raw.elevenlabs;
        let elevenlabs = ElevenLabsConfig {
            api_key_env: non_empty(el.api_key_env).unwrap_or_else(|| "ELEVENLABS_API_KEY".into()),
            keychain_service: non_empty(el.keychain_service)
                .unwrap_or_else(|| "elevenlabs-api-key".into()),
            plan: non_empty(el.plan)
                .map(|p| p.trim().to_lowercase())
                .unwrap_or_else(|| "free".into()),
            concurrency: el.concurrency.unwrap_or(2),
            per_call_credit_limit: el.per_call_credit_limit.unwrap_or(300),
            daily_credit_limit: el.daily_credit_limit.unwrap_or(2000),
            credits_per_second: el.credits_per_second.unwrap_or(10),
            credits_auto_duration: el.credits_auto_duration.unwrap_or(50),
            timeout_s: el.timeout_s.unwrap_or(60),
        };
        for (key, value) in [
            ("concurrency", elevenlabs.concurrency as u64),
            ("credits_per_second", elevenlabs.credits_per_second.into()),
            ("timeout_s", elevenlabs.timeout_s),
        ] {
            if value == 0 {
                return Err(bad(Some("elevenlabs"), key, "must be at least 1".into()));
            }
        }

        let true_peak_dbtp = raw.post.true_peak_dbtp.unwrap_or(-1.0);
        if !(true_peak_dbtp.is_finite() && (-20.0..=0.0).contains(&true_peak_dbtp)) {
            return Err(bad(
                Some("post"),
                "true_peak_dbtp",
                format!("must be between -20 and 0 dBTP (got {true_peak_dbtp})"),
            ));
        }

        Ok(Self {
            config_dir: dir.to_path_buf(),
            cache_root: non_empty_env(CACHE_DIR_ENV)
                .map(|d| expand_tilde(&d))
                .or_else(dirs::cache_dir)
                .unwrap_or_else(std::env::temp_dir),
            data_root: non_empty_env(DATA_DIR_ENV)
                .map(|d| expand_tilde(&d))
                .or_else(dirs::data_dir)
                .unwrap_or_else(std::env::temp_dir),
            allowed_roots,
            default_output_dir,
            default_model: non_empty(raw.default_model).unwrap_or_else(|| DEFAULT_MODEL.into()),
            hf_cache: non_empty(raw.hf_cache)
                .map(|d| resolve(&d))
                .unwrap_or_else(default_hf_cache),
            memory_headroom_mb: raw.memory_headroom_mb.unwrap_or(1000),
            heavy_processes: raw
                .heavy_processes
                .unwrap_or_else(|| DEFAULT_HEAVY.iter().map(|s| s.to_string()).collect()),
            job_ttl_s: raw.job_ttl_s.unwrap_or(3600),
            queue_depth,
            idle_unload_s: raw.idle_unload_s.unwrap_or(600),
            sa3_mlx,
            elevenlabs,
            post: PostConfig { true_peak_dbtp },
        })
    }

    pub fn models_path(&self) -> PathBuf {
        self.config_dir.join("models.toml")
    }

    /// This server's own cache (the embedded worker script is written here, and job scratch folders).
    pub fn app_cache_dir(&self) -> PathBuf {
        self.cache_root.join(APP_DIR)
    }

    /// Scratch folders for backends' temporary files, one per job (removed when the job ends).
    pub fn scratch_dir(&self) -> PathBuf {
        self.app_cache_dir().join("jobs")
    }

    /// The credit ledger: JSON lines (R10).
    pub fn ledger_path(&self) -> PathBuf {
        self.data_root.join(APP_DIR).join("credits.jsonl")
    }

    /// The cross-process lock for local GPU jobs, shared with sister servers (R11).
    pub fn gpu_lock_path(&self) -> PathBuf {
        self.cache_root.join("zavora-mcp").join("local-gpu.lock")
    }
}

/// `env` if set (made absolute against `cwd`), else `<platform>/mcp-sfxgen`, else `~/.config/mcp-sfxgen`.
fn config_dir_from(env: Option<String>, platform: Option<PathBuf>, cwd: &Path) -> PathBuf {
    if let Some(dir) = env {
        return resolve_path(cwd, &dir);
    }
    platform
        .unwrap_or_else(|| home_dir().join(".config"))
        .join(APP_DIR)
}

/// `$HF_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub` (what `huggingface_hub` itself uses).
pub fn default_hf_cache() -> PathBuf {
    hf_cache_from(
        non_empty_env("HF_HUB_CACHE"),
        non_empty_env("HF_HOME"),
        &home_dir(),
    )
}

fn hf_cache_from(hub_cache: Option<String>, hf_home: Option<String>, home: &Path) -> PathBuf {
    if let Some(dir) = hub_cache {
        return expand_tilde(&dir);
    }
    if let Some(dir) = hf_home {
        return expand_tilde(&dir).join("hub");
    }
    home.join(".cache").join("huggingface").join("hub")
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// The user's home directory (the OS's notion of it), else the temp dir.
pub fn home_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(std::env::temp_dir)
}

/// Expand a leading `~`, `~/` or `~\` to the home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        home_dir()
    } else if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        home_dir().join(rest)
    } else {
        PathBuf::from(path)
    }
}

/// Expand `~`, then anchor a relative path at `base`.
pub fn resolve_path(base: &Path, path: &str) -> PathBuf {
    let expanded = expand_tilde(path.trim());
    if expanded.is_absolute() {
        expanded
    } else {
        base.join(expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(text: &str) -> Config {
        Config::from_toml(Path::new("/w/.sfxgen"), text).unwrap()
    }

    fn err(text: &str) -> String {
        Config::from_toml(Path::new("/w/.sfxgen"), text).unwrap_err()
    }

    #[test]
    fn example_config_parses_with_design_values() {
        let c = cfg(CONFIG_EXAMPLE);
        assert_eq!(
            c.allowed_roots,
            ["generated", "nairobi-racer", "sunfall", "lab"].map(|r| Path::new("/w").join(r))
        );
        assert_eq!(c.default_output_dir, Path::new("/w/generated/sfx"));
        assert_eq!(c.default_model, "sa3-small-sfx");
        assert_eq!(c.memory_headroom_mb, 1000);
        assert_eq!(
            c.heavy_processes,
            [
                "UnrealEditor",
                "Blender",
                "VoiceStudio",
                "sd-server",
                "trellis",
                "mcp-meshgen"
            ]
        );
        assert_eq!(
            (c.job_ttl_s, c.queue_depth, c.idle_unload_s),
            (3600, 4, 600)
        );
        assert!(c.sa3_mlx.dir.ends_with("stable-audio-3/optimized/mlx"));
        assert!(c.sa3_mlx.dir.is_absolute(), "~ is expanded");
        assert_eq!(c.sa3_mlx.licence_accepted, None);
        assert!(!c.sa3_mlx.licence_is_accepted());
        assert!(c.sa3_mlx.python_path().starts_with(&c.sa3_mlx.dir));
        let el = &c.elevenlabs;
        assert_eq!(el.api_key_env, "ELEVENLABS_API_KEY");
        assert_eq!(el.keychain_service, "elevenlabs-api-key");
        assert_eq!(el.plan, "free");
        assert!(!el.commercial_use());
        assert_eq!(
            (
                el.concurrency,
                el.per_call_credit_limit,
                el.daily_credit_limit,
                el.credits_per_second,
                el.credits_auto_duration,
                el.timeout_s
            ),
            (2, 300, 2000, 10, 50, 60)
        );
        assert_eq!(c.post.true_peak_dbtp, -1.0);
    }

    #[test]
    fn empty_config_uses_defaults() {
        let c = cfg("");
        assert_eq!(c.default_model, "sa3-small-sfx");
        assert_eq!(c.allowed_roots.len(), 1);
        assert!(c.allowed_roots[0].ends_with("Documents/mcp-sfxgen"));
        assert_eq!(c.default_output_dir, c.allowed_roots[0].join("sfx"));
        assert_eq!(c.queue_depth, 4);
        assert_eq!(c.elevenlabs.per_call_credit_limit, 300);
        assert_eq!(c.post.true_peak_dbtp, -1.0);
    }

    #[test]
    fn paths_resolve_against_the_config_dirs_parent() {
        let c = cfg(
            "allowed_roots = [\"assets\", \"/abs/root\", \"~/sfx\"]\ndefault_output_dir = \"assets/out\"\nhf_cache = \"models/hub\"\n[sa3_mlx]\ndir = \"tools/sa3\"",
        );
        assert_eq!(c.allowed_roots[0], Path::new("/w/assets"));
        assert_eq!(c.allowed_roots[1], Path::new("/abs/root"));
        assert_eq!(c.allowed_roots[2], home_dir().join("sfx"));
        assert_eq!(c.default_output_dir, Path::new("/w/assets/out"));
        assert_eq!(c.hf_cache, Path::new("/w/models/hub"));
        assert_eq!(c.sa3_mlx.dir, Path::new("/w/tools/sa3"));
    }

    #[test]
    fn tilde_expansion() {
        assert_eq!(expand_tilde("~"), home_dir());
        assert_eq!(expand_tilde("~/x/y"), home_dir().join("x/y"));
        assert_eq!(expand_tilde("~\\x"), home_dir().join("x"));
        assert_eq!(expand_tilde("/a/~/b"), Path::new("/a/~/b"));
        assert_eq!(expand_tilde("~user/x"), Path::new("~user/x"));
    }

    #[test]
    fn licence_date_is_parsed() {
        let c = cfg("[sa3_mlx]\nlicence_accepted = \"2026-10-03\"");
        assert_eq!(
            c.sa3_mlx.licence_accepted,
            chrono::NaiveDate::from_ymd_opt(2026, 10, 3)
        );
        assert!(c.sa3_mlx.licence_is_accepted());
        let e = err("# a comment\n[sa3_mlx]\nlicence_accepted = \"yesterday\"");
        assert!(e.contains("sa3_mlx.licence_accepted"), "{e}");
        assert!(e.contains("(line 3)"), "{e}");
    }

    #[test]
    fn plan_is_lowercased_and_only_free_is_non_commercial() {
        assert!(
            !cfg("[elevenlabs]\nplan = \"Free\"")
                .elevenlabs
                .commercial_use()
        );
        assert!(
            cfg("[elevenlabs]\nplan = \"Creator\"")
                .elevenlabs
                .commercial_use()
        );
        assert_eq!(cfg("[elevenlabs]\nplan = \" Pro \"").elevenlabs.plan, "pro");
    }

    #[test]
    fn syntax_and_unknown_key_errors_carry_line_numbers() {
        let e = err("memory_headroom_mb = 1000\nqueue_depth = [\n");
        assert!(e.contains("line 2") || e.contains("line 3"), "{e}");
        let e = err("memory_headroom_mb = 1000\n\n[elevenlabs]\nconcurency = 3\n");
        assert!(e.contains("line 4"), "{e}");
        assert!(e.contains("concurency"), "{e}");
        let e = err("surprise = 1\n");
        assert!(e.contains("line 1") && e.contains("surprise"), "{e}");
        let e = err("[elevenlabs]\nplan = 5\n");
        assert!(e.contains("line 2"), "{e}");
    }

    #[test]
    fn value_errors_name_the_key_and_line() {
        let e = err("allowed_roots = [\"a\"]\ndefault_output_dir = \"elsewhere/out\"\n");
        assert!(
            e.contains("default_output_dir") && e.contains("not inside"),
            "{e}"
        );
        assert!(e.contains("(line 2)"), "{e}");
        let e = err("\n\nqueue_depth = 0\n");
        assert!(e.contains("queue_depth must be at least 1 (line 3)"), "{e}");
        let e = err("[elevenlabs]\nplan = \"free\"\nconcurrency = 0\n");
        assert!(
            e.contains("elevenlabs.concurrency") && e.contains("(line 3)"),
            "{e}"
        );
        let e = err("[post]\ntrue_peak_dbtp = 3.0\n");
        assert!(
            e.contains("post.true_peak_dbtp") && e.contains("(line 2)"),
            "{e}"
        );
    }

    #[test]
    fn locate_respects_tables() {
        let text = "a = 1\n[x]\nkey = 1\n[y]\nkey = 2\n";
        assert_eq!(locate(text, Some("y"), "key"), " (line 5)");
        assert_eq!(locate(text, Some("x"), "key"), " (line 3)");
        assert_eq!(locate(text, None, "a"), " (line 1)");
        assert_eq!(locate(text, None, "key"), "");
        assert_eq!(locate(text, Some("y"), "ke"), "");
    }

    #[test]
    fn config_dir_discovery() {
        let cwd = Path::new("/cwd");
        assert_eq!(
            config_dir_from(Some("rel/dir".into()), Some("/plat".into()), cwd),
            Path::new("/cwd/rel/dir")
        );
        assert_eq!(
            config_dir_from(Some("/abs".into()), Some("/plat".into()), cwd),
            Path::new("/abs")
        );
        assert_eq!(
            config_dir_from(None, Some("/plat".into()), cwd),
            Path::new("/plat/mcp-sfxgen")
        );
        assert_eq!(
            config_dir_from(None, None, cwd),
            home_dir().join(".config/mcp-sfxgen")
        );
    }

    #[test]
    fn hf_cache_discovery() {
        let home = Path::new("/home/u");
        assert_eq!(
            hf_cache_from(Some("/hub".into()), Some("/hfhome".into()), home),
            Path::new("/hub")
        );
        assert_eq!(
            hf_cache_from(None, Some("/hfhome".into()), home),
            Path::new("/hfhome/hub")
        );
        assert_eq!(
            hf_cache_from(None, None, home),
            Path::new("/home/u/.cache/huggingface/hub")
        );
    }

    #[test]
    fn load_or_init_writes_the_examples_once() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join(".sfxgen");
        let c = Config::load_or_init(&dir).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("config.toml")).unwrap(),
            CONFIG_EXAMPLE
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("models.toml")).unwrap(),
            MODELS_EXAMPLE
        );
        assert_eq!(c.models_path(), dir.join("models.toml"));
        // Relative roots anchor at the config dir's parent.
        assert_eq!(c.allowed_roots[0], root.path().join("generated"));
        // An edited file is never overwritten.
        std::fs::write(dir.join("config.toml"), "queue_depth = 7\n").unwrap();
        assert_eq!(Config::load_or_init(&dir).unwrap().queue_depth, 7);
    }

    #[test]
    fn load_or_init_reports_the_file_and_line() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join(".sfxgen");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.toml"), "queue_depth = 4\nbogus = 1\n").unwrap();
        let e = Config::load_or_init(&dir).unwrap_err().to_string();
        assert!(e.contains("config.toml") && e.contains("line 2"), "{e}");
    }

    #[test]
    fn derived_paths() {
        let c = cfg("");
        assert!(c.gpu_lock_path().ends_with("zavora-mcp/local-gpu.lock"));
        assert!(c.app_cache_dir().ends_with("mcp-sfxgen"));
        assert!(c.scratch_dir().ends_with("mcp-sfxgen/jobs"));
        assert!(c.ledger_path().ends_with("mcp-sfxgen/credits.jsonl"));
    }

    #[test]
    fn credentials_hint_names_the_env_var_and_service() {
        let hint = cfg("").elevenlabs.credentials_hint();
        assert!(hint.contains("ELEVENLABS_API_KEY") && hint.contains("elevenlabs-api-key"));
    }
}
