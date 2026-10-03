//! Heavy-process census (design §8; R11): when memory is short, name what to close.
//!
//! A process matches a configured pattern by its name or by its command line (so a `python … trellis …` worker
//! counts), and the built-in `CMDLINE_ALIASES` name our own Python worker "mcp-sfxgen worker".

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde::Serialize;

/// Processes below this are not worth naming.
const MIN_REPORT_MB: u64 = 200;
/// With no configured match, name this many of the largest processes instead.
const FALLBACK_COUNT: usize = 3;
const FALLBACK_MIN_MB: u64 = 1_000;
/// Command lines are cut here; only a fragment is ever matched.
const CMDLINE_MAX: usize = 1_024;

/// Command-line fragments with a name of their own: a Python process running our worker script is
/// "mcp-sfxgen worker", not "python3.12". Checked before the configured patterns.
pub const CMDLINE_ALIASES: [(&str, &str); 1] = [("sa3_mlx_worker", "mcp-sfxgen worker")];

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessInfo {
    pub name: String,
    /// The full command line (space-joined, possibly cut), or empty where it can't be read.
    pub cmdline: String,
    /// Physical footprint on macOS (GPU allocations included), resident size elsewhere.
    pub resident_mb: u64,
}

pub trait ProcessLister: Send + Sync {
    fn processes(&self) -> Vec<ProcessInfo>;
}

/// Real lister backed by `sysinfo`.
pub struct SystemProcesses {
    sys: Mutex<sysinfo::System>,
}

impl Default for SystemProcesses {
    fn default() -> Self {
        Self {
            sys: Mutex::new(sysinfo::System::new()),
        }
    }
}

impl ProcessLister for SystemProcesses {
    fn processes(&self) -> Vec<ProcessInfo> {
        use sysinfo::{ProcessRefreshKind, ProcessesToUpdate};
        let mut sys = self.sys.lock().unwrap_or_else(|e| e.into_inner());
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            true,
            ProcessRefreshKind::nothing()
                .with_memory()
                .with_cmd(sysinfo::UpdateKind::OnlyIfNotSet),
        );
        let own = std::process::id();
        sys.processes()
            .iter()
            .filter(|(pid, _)| pid.as_u32() != own)
            .map(|(pid, p)| ProcessInfo {
                name: p.name().to_string_lossy().into_owned(),
                cmdline: cmdline_of(p.cmd()),
                resident_mb: footprint_bytes(pid.as_u32()).unwrap_or_else(|| p.memory())
                    / (1024 * 1024),
            })
            .collect()
    }
}

/// The argv joined by spaces, cut at `CMDLINE_MAX` characters.
fn cmdline_of(cmd: &[std::ffi::OsString]) -> String {
    let joined = cmd
        .iter()
        .map(|a| a.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    joined.chars().take(CMDLINE_MAX).collect()
}

/// macOS: the physical footprint (Activity Monitor's "Memory"), which counts GPU/Metal allocations that resident size
/// misses on unified memory: Unreal Editor measured 9958 MB footprint against 1627 MB resident. `None` where it can't be
/// read (another user's process) or on other platforms, so the caller falls back to resident size.
#[cfg(target_os = "macos")]
fn footprint_bytes(pid: u32) -> Option<u64> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: `info` is a properly sized, writable rusage_info_v2 for flavor RUSAGE_INFO_V2.
    let rc = unsafe {
        libc::proc_pid_rusage(
            i32::try_from(pid).ok()?,
            libc::RUSAGE_INFO_V2,
            info.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    // SAFETY: proc_pid_rusage filled the struct when it returned 0.
    (rc == 0).then(|| unsafe { info.assume_init() }.ri_phys_footprint)
}

#[cfg(not(target_os = "macos"))]
fn footprint_bytes(_pid: u32) -> Option<u64> {
    None
}

/// Fixed list, for tests.
pub struct FixedProcesses(pub Vec<ProcessInfo>);

impl ProcessLister for FixedProcesses {
    fn processes(&self) -> Vec<ProcessInfo> {
        self.0.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeavyProcess {
    /// The configured name that matched (all its processes summed), or the process name in the fallback.
    pub name: String,
    pub resident_mb: u64,
    pub count: usize,
}

/// Who a process counts as: an alias for its command line, else the first configured pattern its name contains, else
/// the first its command line contains (a name match wins, so `python` running `trellis` still counts as `trellis`
/// only when no pattern names the executable).
fn label_for(p: &ProcessInfo, patterns: &[String]) -> Option<String> {
    let name = p.name.to_lowercase();
    let cmdline = p.cmdline.to_lowercase();
    if let Some((_, label)) = CMDLINE_ALIASES
        .iter()
        .find(|(fragment, _)| cmdline.contains(fragment))
    {
        return Some((*label).to_string());
    }
    let wanted = |text: &str| {
        patterns
            .iter()
            .find(|pat| !pat.is_empty() && text.contains(&pat.to_lowercase()))
            .cloned()
    };
    wanted(&name).or_else(|| wanted(&cmdline))
}

/// Processes matching `patterns` (case-insensitive substring of the name or the command line), summed per pattern,
/// largest first. With no match, the few largest processes over 1 GB, so the message still points somewhere.
pub fn census(processes: &[ProcessInfo], patterns: &[String]) -> Vec<HeavyProcess> {
    let mut matched: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    for p in processes {
        if let Some(label) = label_for(p, patterns) {
            let entry = matched.entry(label).or_default();
            entry.0 += p.resident_mb;
            entry.1 += 1;
        }
    }
    let mut heavy: Vec<HeavyProcess> = matched
        .into_iter()
        .filter(|(_, (mb, _))| *mb >= MIN_REPORT_MB)
        .map(|(name, (resident_mb, count))| HeavyProcess {
            name,
            resident_mb,
            count,
        })
        .collect();
    if heavy.is_empty() {
        heavy = processes
            .iter()
            .filter(|p| p.resident_mb >= FALLBACK_MIN_MB)
            .map(|p| HeavyProcess {
                name: p.name.clone(),
                resident_mb: p.resident_mb,
                count: 1,
            })
            .collect();
        heavy.sort_by_key(|h| std::cmp::Reverse(h.resident_mb));
        heavy.truncate(FALLBACK_COUNT);
    } else {
        heavy.sort_by_key(|h| std::cmp::Reverse(h.resident_mb));
    }
    heavy
}

/// "Close UnrealEditor (6100 MB), ComfyUI (4800 MB)." or an empty string when nothing is worth naming.
pub fn advice(heavy: &[HeavyProcess]) -> String {
    if heavy.is_empty() {
        return String::new();
    }
    let list: Vec<String> = heavy
        .iter()
        .map(|h| {
            if h.count > 1 {
                format!("{} ({} MB in {} processes)", h.name, h.resident_mb, h.count)
            } else {
                format!("{} ({} MB)", h.name, h.resident_mb)
            }
        })
        .collect();
    format!("Close {}.", list.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(name: &str, mb: u64) -> ProcessInfo {
        ProcessInfo {
            name: name.into(),
            cmdline: String::new(),
            resident_mb: mb,
        }
    }

    fn py(cmdline: &str, mb: u64) -> ProcessInfo {
        ProcessInfo {
            name: "python3.12".into(),
            cmdline: cmdline.into(),
            resident_mb: mb,
        }
    }

    fn patterns() -> Vec<String> {
        [
            "UnrealEditor",
            "Blender",
            "ComfyUI",
            "sd-server",
            "mcp-meshgen",
            "trellis",
        ]
        .map(String::from)
        .to_vec()
    }

    #[test]
    fn matches_sums_and_sorts() {
        let procs = [
            p("UnrealEditor", 6100),
            p("UnrealEditor-Cmd", 900),
            p("comfyui-backend", 4800),
            p("Blender", 150),
            p("Safari", 3000),
            p("mcp-meshgen", 40),
        ];
        let heavy = census(&procs, &patterns());
        assert_eq!(
            heavy,
            vec![
                HeavyProcess {
                    name: "UnrealEditor".into(),
                    resident_mb: 7000,
                    count: 2
                },
                HeavyProcess {
                    name: "ComfyUI".into(),
                    resident_mb: 4800,
                    count: 1
                },
            ]
        );
        assert_eq!(
            advice(&heavy),
            "Close UnrealEditor (7000 MB in 2 processes), ComfyUI (4800 MB)."
        );
    }

    #[test]
    fn falls_back_to_the_largest_processes() {
        let procs = [
            p("Safari", 3000),
            p("Xcode", 5000),
            p("Finder", 90),
            p("Mail", 1200),
            p("Slack", 1100),
        ];
        let heavy = census(&procs, &patterns());
        let names: Vec<&str> = heavy.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names, ["Xcode", "Safari", "Mail"]);
    }

    #[test]
    fn nothing_heavy_means_no_advice() {
        let heavy = census(&[p("Finder", 90)], &patterns());
        assert!(heavy.is_empty());
        assert_eq!(advice(&heavy), "");
    }

    #[test]
    fn empty_patterns_never_match_everything() {
        let heavy = census(&[p("Safari", 300)], &[String::new()]);
        assert!(heavy.is_empty(), "{heavy:?}");
    }

    #[test]
    fn a_python_worker_is_named_by_its_command_line() {
        let procs = [
            py(
                "/opt/sa3/.venv/bin/python -u /cache/mcp-sfxgen/worker/sa3_mlx_worker-1a2b3c4d5e6f.py /opt/sa3",
                1700,
            ),
            py("python manage.py runserver", 900),
            p("Safari", 3000),
        ];
        let heavy = census(&procs, &patterns());
        assert_eq!(
            heavy,
            vec![HeavyProcess {
                name: "mcp-sfxgen worker".into(),
                resident_mb: 1700,
                count: 1
            }]
        );
        assert_eq!(advice(&heavy), "Close mcp-sfxgen worker (1700 MB).");
    }

    #[test]
    fn configured_patterns_also_match_command_lines() {
        let procs = [
            py(
                "/venv/bin/python batch_generate.py --model trellis-mac",
                14_000,
            ),
            py("python -m http.server", 300),
            // The name wins over the command line.
            ProcessInfo {
                name: "Blender".into(),
                cmdline: "/Applications/Blender.app/Contents/MacOS/Blender --trellis".into(),
                resident_mb: 800,
            },
        ];
        let heavy = census(&procs, &patterns());
        assert_eq!(
            heavy,
            vec![
                HeavyProcess {
                    name: "trellis".into(),
                    resident_mb: 14_000,
                    count: 1
                },
                HeavyProcess {
                    name: "Blender".into(),
                    resident_mb: 800,
                    count: 1
                },
            ]
        );
    }

    #[test]
    fn alias_processes_sum_into_one_entry() {
        let procs = [
            py("python -u sa3_mlx_worker-aaaa.py /d", 1200),
            py("python -u sa3_mlx_worker-bbbb.py /d", 600),
        ];
        let heavy = census(&procs, &patterns());
        assert_eq!(heavy.len(), 1);
        assert_eq!((heavy[0].resident_mb, heavy[0].count), (1800, 2));
        assert_eq!(
            advice(&heavy),
            "Close mcp-sfxgen worker (1800 MB in 2 processes)."
        );
    }

    /// The real lister reads command lines: a child whose argv names the worker script is seen by it.
    #[cfg(unix)]
    #[test]
    fn system_lister_reads_command_lines() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "sleep 30", "sa3_mlx_worker-testmarker.py"])
            .spawn()
            .unwrap();
        let mut seen = None;
        for _ in 0..50 {
            let procs = SystemProcesses::default().processes();
            seen = procs
                .into_iter()
                .find(|p| p.cmdline.contains("sa3_mlx_worker-testmarker.py"));
            if seen.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
        let seen = seen.expect("the child's command line is visible");
        let labelled = census(
            &[ProcessInfo {
                resident_mb: 1700,
                ..seen
            }],
            &[],
        );
        assert_eq!(labelled[0].name, "mcp-sfxgen worker");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn footprint_is_readable_for_our_own_process() {
        let bytes = footprint_bytes(std::process::id()).expect("own footprint");
        assert!(bytes > 0);
    }

    #[test]
    fn system_lister_sees_processes() {
        let procs = SystemProcesses::default().processes();
        assert!(!procs.is_empty());
        assert!(procs.iter().any(|p| p.resident_mb > 0));
    }
}
