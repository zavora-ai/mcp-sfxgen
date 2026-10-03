//! Paths in and out (design §5, §7; R1, R10, R12), adapted from mcp-meshgen.
//!
//! Outputs are jailed under the configured roots. A call writes one family of files per variation sharing a stem:
//! `<name>_NN.wav` (the processed copy), `<name>_NN.raw.wav` (the backend's output, converted only to WAV) and
//! `<name>_NN.json` (the sidecar), for NN = 01..variations. A name is free only when no file in the folder belongs to
//! its family (`<name>_<digits>.…`), and all variations are reserved together by creating `<name>_NN.wav.partial`
//! exclusively, so two jobs never share a name. Existing files are never overwritten; partials are renamed on success
//! and deleted on cancel and failure.
//!
//! An explicit `output_path` (`.wav`, one variation) is exact instead: `<stem>.wav`, `<stem>.raw.wav`, `<stem>.json`,
//! and an existing family is an `output_exists` error, not a new name.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::config::{Config, expand_tilde};
use crate::error::SfxError;

/// Suffix of files still being written.
pub const PARTIAL: &str = "partial";

/// The longest `name` a caller may give.
pub const MAX_NAME_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub struct PathPolicy {
    /// Relative paths resolve here: the config dir's parent, like the roots.
    pub base: PathBuf,
    pub default_dir: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
}

/// Where a job's files go, before collision handling.
#[derive(Debug, Clone, PartialEq)]
pub struct SfxTarget {
    pub dir: PathBuf,
    pub stem: String,
    /// `<stem>_NN` per variation (a free stem is found), or exactly `<stem>` (one variation, an explicit path).
    pub numbered: bool,
}

/// A reserved stem and every path derived from it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SfxNames {
    pub dir: PathBuf,
    pub stem: String,
    pub variations: u32,
    pub numbered: bool,
}

impl PathPolicy {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            base: cfg
                .config_dir
                .parent()
                .unwrap_or(&cfg.config_dir)
                .to_path_buf(),
            default_dir: cfg.default_output_dir.clone(),
            allowed_roots: cfg.allowed_roots.clone(),
        }
    }

    /// Expand `~` and anchor a relative path at `base`.
    pub fn absolutize(&self, p: &str) -> PathBuf {
        let p = expand_tilde(p.trim());
        if p.is_absolute() {
            p
        } else {
            self.base.join(p)
        }
    }

    /// Where the files should go: `output_path` (a `.wav`, one variation), else `output_dir`, else the default dir.
    /// The stem is `name`, else the slug of the prompt.
    pub fn plan(
        &self,
        output_path: Option<&str>,
        output_dir: Option<&str>,
        name: Option<&str>,
        prompt: &str,
        variations: u32,
    ) -> Result<SfxTarget, SfxError> {
        let target = if let Some(p) = non_empty(output_path) {
            if variations != 1 {
                return Err(SfxError::conflict(
                    format!(
                        "`output_path` names one file but {variations} variations were asked for"
                    ),
                    "use `output_dir` (and `name`) for several variations, or set variations to 1",
                ));
            }
            if non_empty(name).is_some() || non_empty(output_dir).is_some() {
                return Err(SfxError::conflict(
                    "`output_path` can't be combined with `name` or `output_dir`",
                    "give `output_path` alone, or `output_dir` with `name`",
                ));
            }
            let p = self.absolutize(p);
            let is_wav = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
            let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned());
            match (is_wav, stem, p.parent()) {
                (true, Some(stem), Some(dir)) if !stem.is_empty() => {
                    check_name(&stem, "output_path")?;
                    SfxTarget {
                        dir: dir.to_path_buf(),
                        stem,
                        numbered: false,
                    }
                }
                _ => return Err(SfxError::invalid("output_path", "must end in .wav")),
            }
        } else {
            let stem = match non_empty(name) {
                Some(n) => {
                    check_name(n.trim(), "name")?;
                    n.trim().to_string()
                }
                None => slug(prompt),
            };
            SfxTarget {
                dir: non_empty(output_dir)
                    .map(|d| self.absolutize(d))
                    .unwrap_or_else(|| self.default_dir.clone()),
                stem,
                numbered: true,
            }
        };
        self.jail(&target.dir.join(format!("{}.wav", target.stem)))?;
        Ok(target)
    }

    fn jail(&self, target: &Path) -> Result<(), SfxError> {
        let not_allowed = || SfxError::PathNotAllowed {
            path: target.to_path_buf(),
            roots: self.allowed_roots.clone(),
        };
        // `..` resolves lexically on Windows but not on Unix (and can step around symlinks), so
        // reject it outright for identical, safe behaviour everywhere.
        if target.components().any(|c| c == Component::ParentDir) {
            return Err(not_allowed());
        }
        if within_roots(target, &self.allowed_roots) {
            Ok(())
        } else {
            Err(not_allowed())
        }
    }
}

fn non_empty(v: Option<&str>) -> Option<&str> {
    v.filter(|s| !s.trim().is_empty())
}

/// A file stem a caller may choose: ASCII letters, digits, `_` and `-`, at most `MAX_NAME_LEN`.
fn check_name(name: &str, field: &str) -> Result<(), SfxError> {
    if name.is_empty()
        || name.len() > MAX_NAME_LEN
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(SfxError::invalid(
            field,
            format!(
                "must be 1 to {MAX_NAME_LEN} characters of letters, digits, `_` and `-` (got `{name}`)"
            ),
        ));
    }
    Ok(())
}

/// Canonicalize the deepest existing ancestor (resolving symlinks) and append the rest.
/// `..` in the existing prefix is resolved by canonicalize; in the missing remainder it has no
/// file name to peel off, so the walk returns `None` and the path is rejected.
pub(crate) fn resolve_lenient(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        let name = existing.file_name()?.to_owned();
        rest.push(name);
        existing = existing.parent()?.to_path_buf();
    }
    let mut out = existing.canonicalize().ok()?;
    for part in rest.iter().rev() {
        match Path::new(part).components().next() {
            Some(Component::Normal(_)) => out.push(part),
            _ => return None,
        }
    }
    Some(out)
}

/// True if `path` resolves (symlinks followed) strictly inside one of `roots`.
pub(crate) fn within_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let Some(resolved) = resolve_lenient(path) else {
        return false;
    };
    roots
        .iter()
        .filter_map(|root| resolve_lenient(root))
        .any(|root| resolved.starts_with(&root) && resolved != root)
}

/// True if `file` belongs to the family of `stem`: `<stem>_<digits>.…` (numbered) or `<stem>.…` (exact).
fn in_family(file: &str, stem: &str, numbered: bool) -> bool {
    if numbered {
        let Some(rest) = file.strip_prefix(stem).and_then(|r| r.strip_prefix('_')) else {
            return false;
        };
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        digits > 0 && rest[digits..].starts_with('.')
    } else {
        file.strip_prefix(stem).is_some_and(|r| r.starts_with('.'))
    }
}

/// True if nothing in `dir` belongs to the `stem` family.
fn stem_is_free(dir: &Path, stem: &str, numbered: bool) -> bool {
    match std::fs::read_dir(dir) {
        Ok(entries) => !entries
            .filter_map(Result::ok)
            .any(|e| in_family(&e.file_name().to_string_lossy(), stem, numbered)),
        Err(_) => true,
    }
}

/// Reserve the first free stem (`x`, `x-1`, …) for `variations` clips by creating every `<stem>_NN.wav.partial`
/// exclusively, or none: if any marker already exists the ones made so far are removed and the next stem is tried.
/// An exact target (`numbered: false`) is taken as is, or refused with `output_exists`.
pub fn reserve(target: &SfxTarget, variations: u32) -> Result<SfxNames, SfxError> {
    std::fs::create_dir_all(&target.dir).map_err(|e| SfxError::io(&target.dir, e))?;
    let variations = if target.numbered {
        variations.max(1)
    } else {
        1
    };
    for n in 0.. {
        let stem = if n == 0 {
            target.stem.clone()
        } else {
            format!("{}-{n}", target.stem)
        };
        let names = SfxNames {
            dir: target.dir.clone(),
            stem,
            variations,
            numbered: target.numbered,
        };
        if !stem_is_free(&target.dir, &names.stem, target.numbered) {
            if target.numbered {
                continue;
            }
            return Err(SfxError::OutputExists { path: names.wav(1) });
        }
        match names.create_markers() {
            Ok(()) => return Ok(names),
            Err(MarkerError::Taken) if target.numbered => continue,
            Err(MarkerError::Taken) => return Err(SfxError::OutputExists { path: names.wav(1) }),
            Err(MarkerError::Io(path, e)) => return Err(SfxError::io(&path, e)),
        }
    }
    unreachable!("unbounded search always finds a free stem")
}

enum MarkerError {
    Taken,
    Io(PathBuf, std::io::Error),
}

impl SfxNames {
    /// `<stem>_NN` for variation `index` (1-based), or `<stem>` for an exact target.
    pub fn item_stem(&self, index: u32) -> String {
        if self.numbered {
            format!("{}_{index:02}", self.stem)
        } else {
            self.stem.clone()
        }
    }

    fn file(&self, index: u32, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}{suffix}", self.item_stem(index)))
    }

    /// The processed WAV of variation `index` (1-based).
    pub fn wav(&self, index: u32) -> PathBuf {
        self.file(index, ".wav")
    }

    /// The raw backend output, converted only to WAV.
    pub fn raw(&self, index: u32) -> PathBuf {
        self.file(index, ".raw.wav")
    }

    pub fn sidecar(&self, index: u32) -> PathBuf {
        self.file(index, ".json")
    }

    /// `<path>.partial`: where a file is written before it's final.
    pub fn partial(&self, path: &Path) -> PathBuf {
        let mut name: OsString = path.file_name().unwrap_or_default().to_owned();
        name.push(format!(".{PARTIAL}"));
        path.with_file_name(name)
    }

    fn create_markers(&self) -> Result<(), MarkerError> {
        let mut made = Vec::new();
        for i in 1..=self.variations {
            let marker = self.partial(&self.wav(i));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker)
            {
                Ok(_) => made.push(marker),
                Err(e) => {
                    for m in &made {
                        let _ = std::fs::remove_file(m);
                    }
                    return Err(if e.kind() == std::io::ErrorKind::AlreadyExists {
                        MarkerError::Taken
                    } else {
                        MarkerError::Io(marker, e)
                    });
                }
            }
        }
        Ok(())
    }

    /// Rename `<path>.partial` to `<path>`, never replacing an existing file.
    pub fn finalize(&self, path: &Path) -> Result<PathBuf, SfxError> {
        let partial = self.partial(path);
        if path.exists() {
            return Err(SfxError::io(path, "already exists; refusing to overwrite"));
        }
        std::fs::rename(&partial, path).map_err(|e| SfxError::io(&partial, e))?;
        Ok(path.to_path_buf())
    }

    /// Delete every `*.partial` of this family in the folder (cancel, failure, or a finished job's leftovers).
    pub fn remove_partials(&self) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let tail = format!(".{PARTIAL}");
        entries
            .filter_map(Result::ok)
            .filter(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                name.ends_with(&tail)
                    && in_family(&name, &self.stem, self.numbered)
                    // Another job's `<stem>-1_01…` is a different stem; a longer numbered stem can't match here.
                    && self.owns(&name)
            })
            .filter(|e| std::fs::remove_file(e.path()).is_ok())
            .count()
    }

    /// True if `file` is one of this reservation's own files (variation index within range).
    fn owns(&self, file: &str) -> bool {
        if !self.numbered {
            return true;
        }
        let rest = &file[self.stem.len() + 1..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits
            .parse::<u32>()
            .is_ok_and(|n| (1..=self.variations).contains(&n))
    }
}

/// Write the sidecar JSON atomically (`.tmp` + rename). Never overwrites.
pub fn write_sidecar(path: &Path, metadata: &impl Serialize) -> Result<(), SfxError> {
    if path.exists() {
        return Err(SfxError::io(path, "already exists; refusing to overwrite"));
    }
    let json = serde_json::to_vec_pretty(metadata).map_err(|e| SfxError::io(path, e))?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json).map_err(|e| SfxError::io(&tmp, e))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        SfxError::io(path, e)
    })
}

/// File-name-safe slug: lowercase ASCII words joined by `-`, at most 48 chars.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "sfx".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path) -> PathPolicy {
        PathPolicy {
            base: root.to_path_buf(),
            default_dir: root.join("generated/sfx"),
            allowed_roots: vec![root.join("game"), root.join("generated")],
        }
    }

    fn setup() -> (tempfile::TempDir, PathPolicy) {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        std::fs::create_dir_all(dir.path().join("generated")).unwrap();
        std::fs::create_dir_all(dir.path().join("game")).unwrap();
        (dir, p)
    }

    #[test]
    fn default_target_is_the_default_dir_with_a_prompt_slug() {
        let (dir, p) = setup();
        let t = p
            .plan(None, None, None, "Short tinny horn: meep!", 2)
            .unwrap();
        assert_eq!(t.dir, dir.path().join("generated/sfx"));
        assert_eq!(t.stem, "short-tinny-horn-meep");
        assert!(t.numbered);
    }

    #[test]
    fn name_output_dir_and_output_path() {
        let (dir, p) = setup();
        let t = p
            .plan(None, Some("game/audio"), Some("Imara_Meep"), "x", 1)
            .unwrap();
        assert_eq!(t.dir, dir.path().join("game/audio"));
        assert_eq!(t.stem, "Imara_Meep");
        let t = p
            .plan(Some("generated/props/Horn.WAV"), None, None, "x", 1)
            .unwrap();
        assert_eq!(t.dir, dir.path().join("generated/props"));
        assert_eq!(t.stem, "Horn");
        assert!(!t.numbered);
    }

    #[test]
    fn rejects_escapes_and_bad_extensions() {
        let (_dir, p) = setup();
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("x.wav");
        for bad in [
            foreign.to_str().unwrap(),
            "../x.wav",
            "generated/../../x.wav",
            "elsewhere/x.wav",
        ] {
            let err = p.plan(Some(bad), None, None, "x", 1).unwrap_err();
            assert_eq!(err.code(), "path_outside_roots", "{bad}: {err}");
        }
        assert_eq!(
            p.plan(None, Some("../out"), None, "x", 1)
                .unwrap_err()
                .code(),
            "path_outside_roots"
        );
        assert_eq!(
            p.plan(None, Some(outside.path().to_str().unwrap()), None, "x", 1)
                .unwrap_err()
                .code(),
            "path_outside_roots"
        );
        for bad in ["generated/x.mp3", "generated/.wav", "generated/x"] {
            assert_eq!(
                p.plan(Some(bad), None, None, "x", 1).unwrap_err().code(),
                "invalid_params",
                "{bad}"
            );
        }
    }

    #[test]
    fn option_conflicts_are_caller_errors() {
        let (_dir, p) = setup();
        let e = p
            .plan(Some("generated/a.wav"), None, None, "x", 2)
            .unwrap_err();
        assert_eq!(e.code(), "option_conflict");
        assert!(e.to_string().contains("2 variations"), "{e}");
        for (dir, name) in [(Some("generated"), None), (None, Some("n"))] {
            let e = p
                .plan(Some("generated/a.wav"), dir, name, "x", 1)
                .unwrap_err();
            assert_eq!(e.code(), "option_conflict");
        }
    }

    #[test]
    fn names_are_restricted() {
        let (_dir, p) = setup();
        for bad in ["a b", "a/b", "..", "é", &"x".repeat(65)] {
            let e = p.plan(None, None, Some(bad), "x", 1).unwrap_err();
            assert_eq!(e.code(), "invalid_params", "{bad}");
        }
        assert!(p.plan(None, None, Some("SFX_Horn-01"), "x", 1).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let (dir, p) = setup();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("generated/link")).unwrap();
        let err = p
            .plan(None, Some("generated/link"), None, "x", 1)
            .unwrap_err();
        assert_eq!(err.code(), "path_outside_roots");
    }

    fn target(dir: &Path, stem: &str) -> SfxTarget {
        SfxTarget {
            dir: dir.to_path_buf(),
            stem: stem.into(),
            numbered: true,
        }
    }

    #[test]
    fn family_membership() {
        assert!(in_family("horn_01.wav", "horn", true));
        assert!(in_family("horn_01.raw.wav.partial", "horn", true));
        assert!(in_family("horn_12.json", "horn", true));
        assert!(!in_family("horn_loud_01.wav", "horn", true));
        assert!(!in_family("horn-1_01.wav", "horn", true));
        assert!(!in_family("horn_.wav", "horn", true));
        assert!(!in_family("horn_01", "horn", true));
        assert!(in_family("horn.wav", "horn", false));
        assert!(!in_family("horn_01.wav", "horn", false));
        assert!(!in_family("horns.wav", "horn", false));
    }

    #[test]
    fn reserve_takes_every_variation_atomically() {
        let (dir, _p) = setup();
        let out = dir.path().join("generated");
        let a = reserve(&target(&out, "horn"), 4).unwrap();
        assert_eq!(a.stem, "horn");
        assert_eq!(a.variations, 4);
        for i in 1..=4 {
            assert!(out.join(format!("horn_{i:02}.wav.partial")).exists(), "{i}");
        }
        assert!(!out.join("horn_05.wav.partial").exists());
        // The partials alone reserve the name for a second job, however many variations it wants.
        assert_eq!(reserve(&target(&out, "horn"), 1).unwrap().stem, "horn-1");
        // Any family member blocks a stem: a stray sidecar or a raw file, even with a higher index.
        std::fs::write(out.join("horn-2_03.json"), b"{}").unwrap();
        std::fs::write(out.join("horn-3_01.raw.wav"), b"x").unwrap();
        assert_eq!(reserve(&target(&out, "horn"), 2).unwrap().stem, "horn-4");
        // A different stem that merely shares a prefix doesn't collide.
        assert_eq!(reserve(&target(&out, "hor"), 1).unwrap().stem, "hor");
        assert_eq!(
            reserve(&target(&out, "horn_loud"), 1).unwrap().stem,
            "horn_loud"
        );
    }

    #[test]
    fn a_half_taken_family_is_released_and_skipped() {
        let (dir, _p) = setup();
        let out = dir.path().join("generated");
        // Simulate a race: a marker for variation 3 appears after the free-check.
        let names = SfxNames {
            dir: out.clone(),
            stem: "x".into(),
            variations: 4,
            numbered: true,
        };
        std::fs::write(out.join("x_03.wav.partial"), b"").unwrap();
        assert!(matches!(names.create_markers(), Err(MarkerError::Taken)));
        assert!(
            !out.join("x_01.wav.partial").exists(),
            "markers made so far are removed"
        );
        assert!(!out.join("x_02.wav.partial").exists());
        assert!(
            out.join("x_03.wav.partial").exists(),
            "the other job's marker stays"
        );
        // And `reserve` skips that stem.
        assert_eq!(reserve(&target(&out, "x"), 4).unwrap().stem, "x-1");
    }

    #[test]
    fn concurrent_reservations_never_share_a_name() {
        let (dir, _p) = setup();
        let out = dir.path().join("generated");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let t = target(&out, "burst");
                std::thread::spawn(move || reserve(&t, 3).unwrap().stem)
            })
            .collect();
        let mut stems: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        stems.sort();
        stems.dedup();
        assert_eq!(stems.len(), 8, "{stems:?}");
    }

    #[test]
    fn exact_targets_are_taken_or_refused() {
        let (dir, _p) = setup();
        let out = dir.path().join("generated");
        let exact = SfxTarget {
            dir: out.clone(),
            stem: "boom".into(),
            numbered: false,
        };
        let names = reserve(&exact, 1).unwrap();
        assert_eq!(names.wav(1), out.join("boom.wav"));
        assert_eq!(names.raw(1), out.join("boom.raw.wav"));
        assert_eq!(names.sidecar(1), out.join("boom.json"));
        assert!(out.join("boom.wav.partial").exists());
        // The partial reserves it; an existing file does too.
        let e = reserve(&exact, 1).unwrap_err();
        assert_eq!(e.code(), "output_exists");
        assert!(e.to_string().contains("boom.wav"), "{e}");
        std::fs::write(out.join("other.json"), b"{}").unwrap();
        let other = SfxTarget {
            stem: "other".into(),
            ..exact.clone()
        };
        assert_eq!(reserve(&other, 1).unwrap_err().code(), "output_exists");
    }

    #[test]
    fn reserve_creates_the_folder() {
        let (dir, _p) = setup();
        let t = target(&dir.path().join("generated/new/deep"), "m");
        assert_eq!(
            reserve(&t, 1).unwrap().wav(1),
            dir.path().join("generated/new/deep/m_01.wav")
        );
    }

    #[test]
    fn names_finalize_and_cleanup() {
        let (dir, _p) = setup();
        let out = dir.path().join("generated");
        let names = reserve(&target(&out, "b"), 2).unwrap();
        assert_eq!(names.wav(1), out.join("b_01.wav"));
        assert_eq!(names.wav(2), out.join("b_02.wav"));
        assert_eq!(names.raw(2), out.join("b_02.raw.wav"));
        assert_eq!(names.sidecar(1), out.join("b_01.json"));
        assert_eq!(names.item_stem(2), "b_02");
        assert_eq!(
            names.partial(&names.raw(1)),
            out.join("b_01.raw.wav.partial")
        );

        std::fs::write(names.partial(&names.wav(1)), b"RIFF").unwrap();
        assert_eq!(names.finalize(&names.wav(1)).unwrap(), out.join("b_01.wav"));
        assert_eq!(std::fs::read(out.join("b_01.wav")).unwrap(), b"RIFF");
        // Never overwrites.
        std::fs::write(names.partial(&names.wav(1)), b"second").unwrap();
        assert!(names.finalize(&names.wav(1)).is_err());
        assert_eq!(std::fs::read(out.join("b_01.wav")).unwrap(), b"RIFF");

        // Cleanup removes this family's partials only: variation 1's leftover, 2's marker and a raw partial.
        std::fs::write(names.partial(&names.raw(2)), b"o").unwrap();
        std::fs::write(out.join("other_01.wav.partial"), b"keep").unwrap();
        std::fs::write(out.join("b-1_01.wav.partial"), b"keep").unwrap();
        std::fs::write(out.join("b_07.wav.partial"), b"keep: not ours").unwrap();
        assert_eq!(names.remove_partials(), 3);
        assert!(out.join("other_01.wav.partial").exists());
        assert!(out.join("b-1_01.wav.partial").exists());
        assert!(out.join("b_07.wav.partial").exists());
        assert!(out.join("b_01.wav").exists());
    }

    #[test]
    fn sidecar_is_written_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m_01.json");
        write_sidecar(&path, &serde_json::json!({"seed": 3})).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(v["seed"], 3);
        assert!(!dir.path().join("m_01.json.tmp").exists());
        assert!(write_sidecar(&path, &serde_json::json!({})).is_err());
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("  Horn, Gold  "), "horn-gold");
        assert_eq!(slug("日本語"), "sfx");
        assert!(slug(&"word ".repeat(40)).len() <= 48);
    }
}
