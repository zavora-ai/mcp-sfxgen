//! Stands in for `worker/sa3_mlx_worker.py` in backend tests (feature `mock`; never installed).
//!
//! Speaks protocol 1 (see PROTOCOL.md) and writes small stereo float32 tones as raw little-endian interleaved `.f32` files,
//! `<out_dir>/<id>_<seed>.f32`, through a `.partial` file that is renamed at the end, exactly like the real worker.
//! Needs only `serde_json`. `FAKE_WORKER_MODE` picks the behaviour:
//!   ok (default)            hello, then the real event sequence (`loading` on the first job only), then a result
//!   slow                    300 ms per sampler step and per decoded item (for cancel and progress tests)
//!   error:<stage>:<kind>    an explained error at that stage; the worker stays alive for the next request
//!                           stage: request | loading | condition | sample | decode | write
//!                           kind:  bad_request | missing_weights | nan_output | out_of_memory | worker_exception
//!   crash:<stage>           leave a half-written `.partial` file, print a traceback to stderr, exit(3) in that stage
//!   hang                    after `loading` and the first progress events, block forever (timeout and cancel tests)
//!   hang_hello              never say hello, block forever (startup timeout)
//!   missing_weights         instead of hello: error kind missing_weights (stage startup), exit(3)
//!   bad_protocol            say hello with protocol 99
//! Optional environment:
//!   FAKE_WORKER_LOG=<file>  append each received request line to this file
//!   FAKE_WORKER_FRAMES=<n>  frames per clip (default: round(seconds * 44100), the honest size)
//!   FAKE_WORKER_STEP_MS=<n> delay per sampler step and per decoded item (default 5; `slow` sets 300)

use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SAMPLE_RATE: u32 = 44_100;

fn emit(value: Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{value}");
    let _ = out.flush();
}

fn hang() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Stereo sine, left and right in opposite phase, 440 Hz times 1..=5 chosen by the seed.
fn tone(frames: usize, seed: u64) -> Vec<u8> {
    let freq = 440.0 * (1 + seed % 5) as f32;
    let mut bytes = Vec::with_capacity(frames * 8);
    for i in 0..frames {
        let v = 0.5 * (std::f32::consts::TAU * freq * i as f32 / SAMPLE_RATE as f32).sin();
        bytes.extend_from_slice(&v.to_le_bytes());
        bytes.extend_from_slice(&(-v).to_le_bytes());
    }
    bytes
}

struct Mode {
    raw: String,
}

impl Mode {
    /// `error:<stage>:<kind>` for this stage.
    fn error_at(&self, stage: &str) -> Option<String> {
        let rest = self.raw.strip_prefix("error:")?;
        let (s, kind) = rest.split_once(':').unwrap_or((rest, "worker_exception"));
        (s == stage).then(|| kind.to_string())
    }
    fn crash_at(&self, stage: &str) -> bool {
        self.raw.strip_prefix("crash:") == Some(stage)
    }
}

fn main() {
    let mode = Mode {
        raw: std::env::var("FAKE_WORKER_MODE").unwrap_or_else(|_| "ok".into()),
    };
    let log = std::env::var("FAKE_WORKER_LOG").ok();
    match mode.raw.as_str() {
        "hang_hello" => hang(),
        "missing_weights" => {
            emit(
                json!({"event": "error", "id": "", "stage": "startup", "kind": "missing_weights",
                "message": "missing weight file(s) under /fake/models/mlx: t5gemma_f16.npz",
                "hint": "the worker never downloads; list_models shows the exact download command and size"}),
            );
            std::process::exit(3);
        }
        _ => {}
    }
    let protocol = if mode.raw == "bad_protocol" { 99 } else { 1 };
    emit(
        json!({"event": "hello", "protocol": protocol, "pid": std::process::id(), "python": "3.11.14",
        "mlx": "0.32.2", "revision": "da6edc54ddba10bfd79a077102ded687f80e882b", "weights_ok": true,
        "sa3_dir": "/fake"}),
    );
    eprintln!("fake sa3 worker ready (mode {})", mode.raw);

    let step_ms = std::env::var("FAKE_WORKER_STEP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if mode.raw == "slow" { 300 } else { 5 });
    let step_delay = Duration::from_millis(step_ms);
    let frames_override: Option<usize> = std::env::var("FAKE_WORKER_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok());
    let mut loaded = false;

    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        if let Some(path) = &log
            && let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
        {
            let _ = writeln!(f, "{line}");
        }
        let Ok(Value::Object(req)) = serde_json::from_str::<Value>(&line) else {
            emit(
                json!({"event": "error", "id": "", "stage": "protocol", "kind": "bad_request",
                "message": "a request must be a JSON object", "hint": ""}),
            );
            continue;
        };
        let id = req
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        match req.get("op").and_then(Value::as_str) {
            Some("unload") => {
                emit(json!({"event": "unloaded", "id": id}));
                return;
            }
            Some("ping") => emit(json!({"event": "pong", "id": id})),
            Some("generate") => {
                if let Some(partials) =
                    generate(&mode, &req, &id, step_delay, frames_override, &mut loaded)
                {
                    // an error was already emitted; make sure a failed job leaves no output behind
                    for p in partials {
                        let _ = std::fs::remove_file(p);
                    }
                }
            }
            op => emit(
                json!({"event": "error", "id": id, "stage": "protocol", "kind": "bad_request",
                "message": format!("unknown op {op:?}"), "hint": ""}),
            ),
        }
    }
}

/// Runs one generate request. Returns Some(files to clean up) when the job ended in an error event, None after a result.
fn generate(
    mode: &Mode,
    req: &serde_json::Map<String, Value>,
    id: &str,
    step_delay: Duration,
    frames_override: Option<usize>,
    loaded: &mut bool,
) -> Option<Vec<PathBuf>> {
    let started = Instant::now();
    let fail = |stage: &str, kind: &str, message: String| {
        emit(
            json!({"event": "error", "id": id, "stage": stage, "kind": kind, "message": message,
            "hint": "fake hint"}),
        );
    };
    let injected = |stage: &str| -> Option<String> { mode.error_at(stage) };

    // Request validation (a small copy of the real worker's rules).
    let prompt = req.get("prompt").and_then(Value::as_str).unwrap_or("");
    let seconds = req.get("seconds").and_then(Value::as_f64).unwrap_or(0.0);
    let steps = req.get("steps").and_then(Value::as_u64).unwrap_or(8).max(1);
    let seeds: Vec<u64> = req
        .get("seeds")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    let out_dir = req.get("out_dir").and_then(Value::as_str).unwrap_or("");
    let problem = if !is_valid_id(id) {
        Some("id must be a string of 1-64 characters from A-Z a-z 0-9 . _ -")
    } else if prompt.trim().is_empty() {
        Some("prompt must be a non-empty string")
    } else if !(0.1..=120.0).contains(&seconds) {
        Some("seconds must be a number from 0.1 to 120")
    } else if seeds.is_empty() || seeds.len() > 8 {
        Some("seeds must be a list of 1 to 8 integers")
    } else if !Path::new(out_dir).is_dir() {
        Some("out_dir must be an existing directory")
    } else {
        None
    };
    if let Some(kind) = injected("request") {
        fail("request", &kind, format!("fake {kind}"));
        return Some(vec![]);
    }
    if let Some(message) = problem {
        fail("request", "bad_request", message.to_string());
        return Some(vec![]);
    }

    let mut timings = serde_json::Map::new();
    for key in ["load", "condition", "sample", "decode"] {
        timings.insert(key.into(), 0.0.into());
    }

    // loading (cold only)
    if !*loaded {
        emit(json!({"event": "loading", "id": id}));
        if mode.crash_at("loading") {
            eprintln!("Traceback (most recent call last):\n  fake crash in loading");
            std::process::exit(3);
        }
        if let Some(kind) = injected("loading") {
            fail("loading", &kind, format!("fake {kind}"));
            return Some(vec![]);
        }
        std::thread::sleep(step_delay);
        timings.insert("load".into(), 0.01.into());
        *loaded = true;
    }

    let progress = |stage: &str, step: u64, total: u64, item: Option<usize>| {
        emit(
            json!({"event": "progress", "id": id, "stage": stage, "step": step, "total": total, "item": item}),
        );
    };

    // condition
    progress("condition", 0, 1, None);
    if mode.crash_at("condition") {
        eprintln!("Traceback (most recent call last):\n  fake crash in condition");
        std::process::exit(3);
    }
    if let Some(kind) = injected("condition") {
        fail("condition", &kind, format!("fake {kind}"));
        return Some(vec![]);
    }
    timings.insert("condition".into(), 0.01.into());

    // sample
    progress("sample", 0, steps, None);
    if mode.raw == "hang" {
        hang();
    }
    for step in 1..=steps {
        std::thread::sleep(step_delay);
        progress("sample", step, steps, None);
        if step == 1 {
            if mode.crash_at("sample") {
                eprintln!("Traceback (most recent call last):\n  fake crash in sample");
                std::process::exit(3);
            }
            if let Some(kind) = injected("sample") {
                fail("sample", &kind, format!("fake {kind}"));
                return Some(vec![]);
            }
        }
    }
    timings.insert(
        "sample".into(),
        (step_delay.as_secs_f64() * steps as f64).into(),
    );

    // decode
    let frames = frames_override.unwrap_or((seconds * SAMPLE_RATE as f64).round() as usize);
    for item in 0..seeds.len() {
        progress("decode", item as u64, seeds.len() as u64, Some(item));
        std::thread::sleep(step_delay);
        if item == 0 {
            if mode.crash_at("decode") {
                eprintln!("Traceback (most recent call last):\n  fake crash in decode");
                std::process::exit(3);
            }
            if let Some(kind) = injected("decode") {
                fail("decode", &kind, format!("fake {kind}"));
                return Some(vec![]);
            }
        }
    }
    timings.insert("decode".into(), 0.01.into());

    // write: <out_dir>/<id>_<seed>.f32.partial, renamed when complete
    let mut files: Vec<PathBuf> = Vec::new();
    let mut partials: Vec<PathBuf> = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        let final_path = Path::new(out_dir).join(format!("{id}_{seed}.f32"));
        let partial = PathBuf::from(format!("{}.partial", final_path.display()));
        if i == 0 && mode.crash_at("write") {
            let _ = std::fs::write(&partial, b"half a clip");
            eprintln!("Traceback (most recent call last):\n  fake crash in write");
            std::process::exit(3);
        }
        if let Some(kind) = injected("write") {
            fail("write", &kind, format!("fake {kind}"));
            partials.extend(files);
            return Some(partials);
        }
        if let Err(e) = std::fs::write(&partial, tone(frames, *seed)) {
            fail("write", "worker_exception", format!("OSError: {e}"));
            partials.push(partial);
            partials.extend(files);
            return Some(partials);
        }
        if let Err(e) = std::fs::rename(&partial, &final_path) {
            fail("write", "worker_exception", format!("OSError: {e}"));
            partials.push(partial);
            partials.extend(files);
            return Some(partials);
        }
        files.push(final_path);
    }

    timings.insert("total".into(), started.elapsed().as_secs_f64().into());
    emit(
        json!({"event": "result", "id": id, "sample_rate": SAMPLE_RATE, "channels": 2, "frames": frames,
        "files": files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "seeds": seeds, "timings": timings, "peak_mlx_mb": 2400.0, "active_mlx_mb": 1600.0,
        "peak_rss_mb": 1700.0}),
    );
    None
}
