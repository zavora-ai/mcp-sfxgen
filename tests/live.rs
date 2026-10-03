//! Live tests (design §9, task 14): the real binary against the real backends. Never run by default.
//!
//! - `live_sa3`: needs Apple Silicon, the Stable Audio 3 MLX venv and its weights (`MCP_SFXGEN_SA3_DIR` overrides the
//!   default `~/Developer/unreal_engine/tools/stable-audio-3/optimized/mlx`). Free; about 3.7 GB of memory.
//! - `live_elevenlabs`: needs `ELEVENLABS_API_KEY`. **Spends 10 credits** (one 1-second clip).
//!
//! Run: `cargo test --release --test live -- --ignored --nocapture`

use std::path::PathBuf;
use std::time::Instant;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientCapabilities, ClientInfo, Implementation,
    ProtocolVersion,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{Value, json};

const MEEP: &str = "A thin, feeble, short car horn beep from a small, tired old compact car. One quick meep. Dry, close, no music.";

#[derive(Clone)]
struct Client;

impl ClientHandler for Client {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("live-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info
    }
}

struct Env {
    dir: tempfile::TempDir,
    cfg: PathBuf,
    root: PathBuf,
}

/// A temp workspace with the shipped model registry, the real Stable Audio dir and a licence date (test only).
fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().canonicalize().unwrap();
    let cfg = base.join(".sfxgen");
    let root = base.join("game");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(&root).unwrap();
    let sa3 = std::env::var("MCP_SFXGEN_SA3_DIR")
        .unwrap_or_else(|_| "~/Developer/unreal_engine/tools/stable-audio-3/optimized/mlx".into());
    std::fs::write(
        cfg.join("config.toml"),
        format!(
            "allowed_roots = [\"game\"]\ndefault_output_dir = \"game/sfx\"\ndefault_model = \"sa3-small-sfx\"\n\n\
             [sa3_mlx]\ndir = {sa3:?}\nlicence_accepted = \"2026-10-03\"\n\n\
             [elevenlabs]\nplan = \"free\"\nper_call_credit_limit = 20\ndaily_credit_limit = 100\n"
        ),
    )
    .unwrap();
    std::fs::write(cfg.join("models.toml"), mcp_sfxgen::config::MODELS_EXAMPLE).unwrap();
    Env { dir, cfg, root }
}

async fn spawn(env: &Env) -> RunningService<RoleClient, Client> {
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcp-sfxgen"));
    cmd.env("MCP_SFXGEN_CONFIG_DIR", &env.cfg)
        .env("MCP_SFXGEN_DATA_DIR", env.dir.path().join("data"))
        .env("RUST_LOG", "warn")
        .current_dir(std::env::temp_dir());
    Client
        .serve(TokioChildProcess::new(cmd).unwrap())
        .await
        .unwrap()
}

async fn call(
    svc: &RunningService<RoleClient, Client>,
    name: &'static str,
    args: Value,
) -> CallToolResult {
    svc.peer()
        .call_tool(
            CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone()),
        )
        .await
        .unwrap()
}

fn sc(r: &CallToolResult) -> Value {
    r.structured_content.clone().expect("structured content")
}

fn assert_ok(r: &CallToolResult) -> Value {
    let v = sc(r);
    assert_ne!(r.is_error, Some(true), "tool error: {v:#}");
    v
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs Apple Silicon, the Stable Audio 3 MLX venv and weights"]
async fn live_sa3() {
    let env = env();
    let svc = spawn(&env).await;

    let models = sc(&call(&svc, "list_models", json!({})).await);
    let sa3 = models["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "sa3-small-sfx")
        .unwrap();
    assert_eq!(sa3["status"], "ready", "{sa3:#}");

    // Cold: 4 variations of the meep.
    let t = Instant::now();
    let v = assert_ok(
        &call(
            &svc,
            "generate_sfx",
            json!({"prompt": MEEP, "duration_s": 1.0, "variations": 4, "name": "meep"}),
        )
        .await,
    );
    let cold = t.elapsed().as_secs_f64();
    let files = v["files"].as_array().unwrap();
    assert_eq!(files.len(), 4);
    for f in files {
        assert!(
            (f["duration_s"].as_f64().unwrap() - 1.0).abs() < 0.001,
            "{f:#}"
        );
        assert!(f["true_peak_dbtp"].as_f64().unwrap() <= -0.99, "{f:#}");
        assert!(PathBuf::from(f["path"].as_str().unwrap()).exists());
        assert!(PathBuf::from(f["raw_path"].as_str().unwrap()).exists());
        assert!(PathBuf::from(f["sidecar"].as_str().unwrap()).exists());
    }
    assert_eq!(v["commercial_use"], "conditional");

    // Warm: one variation.
    let t = Instant::now();
    assert_ok(
        &call(
            &svc,
            "generate_sfx",
            json!({"prompt": MEEP, "duration_s": 1.0, "name": "meep_warm"}),
        )
        .await,
    );
    let warm = t.elapsed().as_secs_f64();

    // A loop, mono at 48 kHz / 24-bit.
    let v = assert_ok(
        &call(
            &svc,
            "generate_sfx",
            json!({"prompt": "A small four-cylinder petrol engine idling steadily, exterior, close to the exhaust. Constant, no revving.",
                   "duration_s": 1.5, "loop": true, "name": "idle",
                   "post": {"mono": true, "sample_rate": 48000, "bit_depth": 24}}),
        )
        .await,
    );
    let f = &v["files"][0];
    assert_eq!(f["channels"], 1);
    assert_eq!(f["sample_rate"], 48000);
    assert_eq!(f["bit_depth"], 24);
    assert!(
        (f["duration_s"].as_f64().unwrap() - 1.5).abs() < 0.001,
        "{f:#}"
    );
    let seam = f["seam"]["wrap_ratio"].as_f64().unwrap();

    println!(
        "live_sa3: cold 4×1 s {cold:.2} s, warm 1×1 s {warm:.2} s, loop seam wrap_ratio {seam:.3}"
    );
    println!("live_sa3: files in {}", env.root.display());
    assert!(warm < 10.0, "warm call took {warm:.2} s");
    assert!(seam < 3.0, "loop seam wrap_ratio {seam}");

    let unloaded = sc(&call(&svc, "unload_models", json!({})).await);
    assert!(
        unloaded["unloaded"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m == "sa3-small-sfx"),
        "{unloaded:#}"
    );
    svc.cancel().await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "spends 10 ElevenLabs credits; needs ELEVENLABS_API_KEY"]
async fn live_elevenlabs() {
    assert!(
        std::env::var("ELEVENLABS_API_KEY").is_ok_and(|k| !k.is_empty()),
        "set ELEVENLABS_API_KEY to run this test"
    );
    let env = env();
    let svc = spawn(&env).await;

    let v = assert_ok(
        &call(&svc, "generate_sfx", json!({"prompt": MEEP, "model": "elevenlabs-sfx-v2", "duration_s": 1.0, "name": "meep_el"})).await,
    );
    assert_eq!(v["credits"], 10, "{v:#}");
    assert_eq!(
        v["commercial_use"], "no",
        "Free plan output is non-commercial"
    );
    let f = &v["files"][0];
    assert!(
        (f["duration_s"].as_f64().unwrap() - 1.0).abs() < 0.001,
        "{f:#}"
    );
    assert!(f["true_peak_dbtp"].as_f64().unwrap() <= -0.99, "{f:#}");

    // Over the per-call limit (20): refused before anything is sent.
    let refused = call(
        &svc,
        "generate_sfx",
        json!({"prompt": MEEP, "model": "elevenlabs-sfx-v2", "duration_s": 3.0}),
    )
    .await;
    assert_eq!(refused.is_error, Some(true));
    assert_eq!(sc(&refused)["error"]["code"], "credit_limit");

    let acct = sc(&call(&svc, "account_status", json!({})).await);
    assert_eq!(acct["credentials_present"], true);
    assert_eq!(acct["ledger"]["today"], 10, "{acct:#}");
    println!("live_elevenlabs: {v:#}\naccount: {acct:#}");
    svc.cancel().await.unwrap();
}
