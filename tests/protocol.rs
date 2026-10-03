//! Protocol tests (design §9): spawn the real binary over stdio with the mock backends and drive it as MCP 2025-11-25
//! (legacy) and 2026-07-28 (Tasks) clients.
#![cfg(feature = "mock")]

use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientCapabilities, ClientInfo,
    GetTaskParams, Implementation, ProtocolVersion, TaskPayload,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{Value, json};

#[derive(Clone)]
struct LegacyClient;

impl ClientHandler for LegacyClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("legacy-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info
    }
}

#[derive(Clone)]
struct TaskClient;

impl ClientHandler for TaskClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::builder().enable_tasks().build(),
            Implementation::new("task-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2026_07_28;
        info
    }
}

struct Env {
    dir: tempfile::TempDir,
    cfg: PathBuf,
    root: PathBuf,
}

impl Env {
    fn out(&self, sub: &str) -> PathBuf {
        self.root.join(sub)
    }

    fn ledger(&self) -> PathBuf {
        self.dir.path().join("data/mcp-sfxgen/credits.jsonl")
    }
}

/// A temp config: one allowed root, the mock models, ElevenLabs limits low enough to test the guard.
fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().canonicalize().unwrap();
    let cfg = base.join(".sfxgen");
    let root = base.join("game");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        cfg.join("config.toml"),
        format!(
            "allowed_roots = [\"game\"]\ndefault_output_dir = \"game/sfx\"\ndefault_model = \"mock-local\"\n\
             memory_headroom_mb = 0\nhf_cache = {hf:?}\nheavy_processes = []\n\n\
             [elevenlabs]\napi_key_env = \"MCP_SFXGEN_TEST_NO_SUCH_KEY\"\nkeychain_service = \"mcp-sfxgen-test-no-such-item\"\n\
             per_call_credit_limit = 45\ndaily_credit_limit = 60\n",
            hf = base.join("hf").display().to_string(),
        ),
    )
    .unwrap();
    std::fs::write(
        cfg.join("models.toml"),
        mcp_sfxgen::backend::mock::MOCK_MODELS,
    )
    .unwrap();
    Env { dir, cfg, root }
}

async fn spawn_with<C: ClientHandler>(
    client: C,
    env: &Env,
    extra: &[(&str, &str)],
) -> RunningService<RoleClient, C> {
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcp-sfxgen"));
    cmd.env("MCP_SFXGEN_CONFIG_DIR", &env.cfg)
        .env("MCP_SFXGEN_CACHE_DIR", env.dir.path().join("cache"))
        .env("MCP_SFXGEN_DATA_DIR", env.dir.path().join("data"))
        .env("MCP_SFXGEN_MOCK_DELAY_MS", "20")
        .env("RUST_LOG", "warn")
        .current_dir(std::env::temp_dir());
    for (k, v) in extra {
        cmd.env(k, v);
    }
    client
        .serve(TokioChildProcess::new(cmd).unwrap())
        .await
        .unwrap()
}

async fn spawn<C: ClientHandler>(client: C, env: &Env) -> RunningService<RoleClient, C> {
    spawn_with(client, env, &[]).await
}

fn call(name: &'static str, v: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(name).with_arguments(v.as_object().unwrap().clone())
}

fn sc(result: &CallToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("structured content")
}

/// Every file under `dir`, recursively, as names relative to it (empty when it doesn't exist).
fn files(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.filter_map(Result::ok) {
            if e.path().is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path().strip_prefix(dir).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}

async fn poll_until_terminal<C: ClientHandler>(
    client: &RunningService<RoleClient, C>,
    job_id: &str,
) -> CallToolResult {
    loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        let status = sc(&r)["status"].as_str().unwrap().to_string();
        if !matches!(status.as_str(), "queued" | "running") {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
}

#[tokio::test]
async fn legacy_client_lists_typed_tools_and_generates_files() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    // tools/list matches the manifest, every tool has a typed output schema, and the annotations are right.
    let manifest = adk_mcp_sdk::ServerManifest::from_toml(mcp_sfxgen::MANIFEST_TOML).unwrap();
    let mut want: Vec<_> = manifest.tools.iter().map(|t| t.name.clone()).collect();
    want.sort();
    let tools = client.list_tools(None).await.unwrap().tools;
    let mut got: Vec<_> = tools.iter().map(|t| t.name.to_string()).collect();
    got.sort();
    assert_eq!(got, want);
    for tool in &tools {
        let json = serde_json::to_value(tool).unwrap();
        let schema = &json["outputSchema"];
        assert_eq!(schema["type"], "object", "{}: {schema}", tool.name);
        assert_ne!(
            schema["additionalProperties"],
            json!(true),
            "{} still has the generic placeholder schema",
            tool.name
        );
        assert!(
            schema.get("properties").is_some() || schema.get("anyOf").is_some(),
            "{}: {schema}",
            tool.name
        );
    }
    let tool =
        |name: &str| serde_json::to_value(tools.iter().find(|t| t.name == name).unwrap()).unwrap();
    let gen_json = tool("generate_sfx");
    assert_eq!(gen_json["annotations"]["readOnlyHint"], false);
    assert_eq!(gen_json["annotations"]["openWorldHint"], true);
    assert_eq!(gen_json["annotations"]["destructiveHint"], false);
    assert_eq!(tool("list_models")["annotations"]["readOnlyHint"], true);
    assert_eq!(tool("list_models")["annotations"]["openWorldHint"], false);
    assert_eq!(tool("get_job")["annotations"]["readOnlyHint"], true);
    assert_eq!(tool("account_status")["annotations"]["readOnlyHint"], true);
    assert_eq!(tool("cancel_job")["annotations"]["idempotentHint"], true);
    // The input schema documents the request.
    let props = &gen_json["inputSchema"]["properties"];
    assert!(props["prompt"]["description"].as_str().unwrap().len() > 20);
    assert_eq!(props["variations"]["maximum"], 4);
    assert!(props["post"].is_object());
    assert_eq!(gen_json["inputSchema"]["required"], json!(["prompt"]));

    let result = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "a cartoon meep", "variations": 2, "duration_s": 1.0, "seed": 100,
                   "output_dir": "game/audio", "name": "meep"}),
        ))
        .await
        .unwrap();
    assert_eq!(result.is_error, Some(false));
    let v = sc(&result);
    assert_eq!(v["model"], "mock-local");
    assert_eq!(v["runs"], "local");
    assert_eq!(v["commercial_use"], "yes");
    assert_eq!(v["params"]["seed"], 100);
    assert_eq!(v["files"].as_array().unwrap().len(), 2);
    let first = &v["files"][0];
    assert_eq!(
        first["path"],
        env.out("audio/meep_01.wav").display().to_string()
    );
    assert_eq!(first["seed"], 100);
    assert_eq!(v["files"][1]["seed"], 101);
    assert!(first["true_peak_dbtp"].as_f64().unwrap() <= -0.99);
    assert_eq!(
        files(&env.out("audio")),
        [
            "meep_01.json",
            "meep_01.raw.wav",
            "meep_01.wav",
            "meep_02.json",
            "meep_02.raw.wav",
            "meep_02.wav"
        ]
    );
    let sidecar: Value =
        serde_json::from_slice(&std::fs::read(env.out("audio/meep_01.json")).unwrap()).unwrap();
    assert_eq!(sidecar["file"], "meep_01.wav");
    assert_eq!(sidecar["prompt"], "a cartoon meep");
    assert_eq!(sidecar["batch"]["size"], 2);
    assert!(
        sidecar["generator"]
            .as_str()
            .unwrap()
            .starts_with("mcp-sfxgen ")
    );
    assert!(sidecar["peaks"]["data"].as_str().unwrap().len() > 100);
    // A text summary, then one resource_link per file.
    let summary = result.content[0].as_text().expect("text summary");
    assert!(
        summary.text.starts_with("Made 2 sounds with mock-local"),
        "{}",
        summary.text
    );
    let links: Vec<_> = result
        .content
        .iter()
        .filter_map(|c| c.as_resource_link())
        .collect();
    assert_eq!(links.len(), 2);
    assert!(links[0].uri.starts_with("file:///") && links[0].uri.ends_with("/meep_01.wav"));
    assert_eq!(links[0].mime_type.as_deref(), Some("audio/wav"));
    assert!(
        files(&env.root).iter().all(|f| !f.ends_with(".partial")),
        "{:?}",
        files(&env.root)
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn task_client_gets_a_task_with_stage_status() {
    let env = env();
    let client = spawn(TaskClient, &env).await;
    let response = client
        .call_tool_once(call(
            "generate_sfx",
            json!({"prompt": "a long hum", "duration_s": 2.0, "variations": 3}),
        ))
        .await
        .unwrap();
    let created = match response {
        CallToolResponse::Task(created) => created,
        other => panic!("expected a task, got {other:?}"),
    };
    assert_eq!(created.task.ttl_ms, Some(3_600_000));
    let mut messages = Vec::new();
    let result = loop {
        let task = client
            .peer()
            .get_task(GetTaskParams::new(created.task.task_id.clone()))
            .await
            .unwrap()
            .task;
        if let Some(m) = &task.task.status_message {
            messages.push(m.clone());
        }
        if task.status().is_terminal() {
            match task.payload {
                TaskPayload::Completed { result } => break result,
                other => panic!("unexpected terminal payload: {other:?}"),
            }
        }
        tokio::time::sleep(Duration::from_millis(15)).await;
    };
    assert_eq!(
        result["structuredContent"]["files"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(result["isError"], false);
    assert!(
        messages.iter().any(|m| m.starts_with("generating ")),
        "status messages seen: {messages:?}"
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn no_wait_then_poll_get_job_on_both_protocols() {
    let env = env();
    for legacy in [true, false] {
        let args = json!({"prompt": "a click", "duration_s": 1.0, "wait": false,
                          "output_dir": if legacy { "game/legacy" } else { "game/tasks" }});
        let done = if legacy {
            let client = spawn(LegacyClient, &env).await;
            let started = client.call_tool(call("generate_sfx", args)).await.unwrap();
            let v = sc(&started);
            assert!(v["hint"].as_str().unwrap().contains("get_job"));
            let id = v["job_id"].as_str().unwrap().to_string();
            let r = poll_until_terminal(&client, &id).await;
            client.cancel().await.unwrap();
            r
        } else {
            // A Tasks client gets a task even with wait=false; it completes at once with the job_id, and get_job (a
            // plain tool) works the same on this protocol.
            let client = spawn(TaskClient, &env).await;
            let created = match client
                .call_tool_once(call("generate_sfx", args))
                .await
                .unwrap()
            {
                CallToolResponse::Task(created) => created,
                other => panic!("expected a task, got {other:?}"),
            };
            let payload = loop {
                let task = client
                    .peer()
                    .get_task(GetTaskParams::new(created.task.task_id.clone()))
                    .await
                    .unwrap()
                    .task;
                if task.status().is_terminal() {
                    match task.payload {
                        TaskPayload::Completed { result } => break result,
                        other => panic!("unexpected terminal payload: {other:?}"),
                    }
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            };
            let id = payload["structuredContent"]["job_id"]
                .as_str()
                .unwrap()
                .to_string();
            let r = poll_until_terminal(&client, &id).await;
            client.cancel().await.unwrap();
            r
        };
        assert_eq!(sc(&done)["status"], "done", "legacy={legacy}");
        assert_eq!(sc(&done)["lane"], "local");
        let path = sc(&done)["result"]["files"][0]["path"].as_str().unwrap();
        assert!(path.ends_with(".wav") && Path::new(path).exists(), "{path}");
    }
}

#[tokio::test]
async fn cancel_mid_job_leaves_no_files() {
    let env = env();
    let client = spawn_with(LegacyClient, &env, &[("MCP_SFXGEN_MOCK_DELAY_MS", "150")]).await;
    let queued = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "a slow one", "variations": 4, "wait": false, "output_dir": "game/slow"}),
        ))
        .await
        .unwrap();
    let job_id = sc(&queued)["job_id"].as_str().unwrap().to_string();
    loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        if sc(&r)["progress_text"]
            .as_str()
            .is_some_and(|p| p.starts_with("generating"))
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        files(&env.out("slow"))
            .iter()
            .any(|f| f.ends_with(".partial")),
        "names are reserved while running: {:?}",
        files(&env.out("slow"))
    );
    client
        .call_tool(call("cancel_job", json!({"job_id": job_id})))
        .await
        .unwrap();
    let r = poll_until_terminal(&client, &job_id).await;
    assert_eq!(sc(&r)["status"], "cancelled");
    assert!(
        files(&env.out("slow")).is_empty(),
        "{:?}",
        files(&env.out("slow"))
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn caller_errors_come_back_as_is_error_results_with_a_fix() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;
    let outside = tempfile::tempdir().unwrap();
    let bad = |args: Value, code: &'static str| {
        let client = &client;
        async move {
            let r = client
                .call_tool(call("generate_sfx", args.clone()))
                .await
                .unwrap_or_else(|e| {
                    panic!("{args} should be an isError result, not a protocol error: {e}")
                });
            assert_eq!(r.is_error, Some(true), "{args}");
            let v = sc(&r);
            assert_eq!(v["error"]["code"], code, "{args}: {v}");
            assert!(!v["error"]["message"].as_str().unwrap().is_empty());
            assert!(!v["error"]["fix"].as_str().unwrap().is_empty(), "{args}");
        }
    };
    bad(json!({"prompt": "  "}), "invalid_params").await;
    bad(json!({"prompt": "x", "variations": 5}), "invalid_params").await;
    bad(
        json!({"prompt": "x", "duration_s": 60.0}),
        "duration_out_of_range",
    )
    .await;
    bad(json!({"prompt": "x", "model": "nope"}), "unknown_model").await;
    bad(
        json!({"prompt": "x", "loop": true, "post": {"fade_in_ms": 10}}),
        "option_conflict",
    )
    .await;
    bad(
        json!({"prompt": "x", "output_path": "game/a.wav", "variations": 2}),
        "option_conflict",
    )
    .await;
    bad(
        json!({"prompt": "x", "output_dir": outside.path().display().to_string()}),
        "path_outside_roots",
    )
    .await;
    bad(
        json!({"prompt": "x", "post": {"sample_rate": 22050}}),
        "invalid_params",
    )
    .await;
    // The same exact path twice: output_exists.
    let ok = json!({"prompt": "x", "output_path": "game/once.wav", "duration_s": 1.0});
    client
        .call_tool(call("generate_sfx", ok.clone()))
        .await
        .unwrap();
    bad(ok, "output_exists").await;
    // Unknown job ids too.
    let r = client
        .call_tool(call("get_job", json!({"job_id": "job_nope"})))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    assert_eq!(sc(&r)["error"]["code"], "unknown_job");
    // A cloud call with the model picking the length is priced at the flat 50 credits, over this config's 45.
    let r = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "x", "model": "mock-cloud"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    assert_eq!(sc(&r)["error"]["code"], "credit_limit");
    assert!(
        sc(&r)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("50 credits")
    );
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn backend_failures_are_protocol_errors_with_a_code_and_leave_no_files() {
    let env = env();
    let client = spawn_with(
        LegacyClient,
        &env,
        &[("MCP_SFXGEN_MOCK_FAIL", "generating:worker_crashed")],
    )
    .await;
    let err = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "x", "output_dir": "game/fail", "variations": 2}),
        ))
        .await
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("worker_crashed"), "{text}");
    assert!(
        files(&env.out("fail")).is_empty(),
        "{:?}",
        files(&env.out("fail"))
    );
    // The failure was injected once: the next call works.
    let ok = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "x", "output_dir": "game/fail"}),
        ))
        .await
        .unwrap();
    assert_eq!(ok.is_error, Some(false));
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn a_credit_limit_refusal_sends_nothing_and_spend_reaches_the_ledger() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;
    // 3 x 2 s = 60 credits, over the per-call limit of 45 (a temp config; the real ledger is never touched).
    let r = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "horn", "model": "mock-cloud", "variations": 3, "duration_s": 2.0,
                   "output_dir": "game/cloud"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    let v = sc(&r);
    assert_eq!(v["error"]["code"], "credit_limit");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("60"),
        "{v}"
    );
    assert!(
        v["error"]["fix"]
            .as_str()
            .unwrap()
            .contains("per_call_credit_limit"),
        "{v}"
    );
    assert!(
        files(&env.out("cloud")).is_empty(),
        "nothing reserved or written"
    );
    assert!(!env.ledger().exists(), "nothing spent");
    let models = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    assert_eq!(sc(&models)["jobs_busy"], 0, "no job was queued");

    // 2 x 2 s = 40 credits passes and is recorded; another 40 would pass the daily limit of 60.
    let r = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "horn", "model": "mock-cloud", "variations": 2, "duration_s": 2.0,
                   "output_dir": "game/cloud"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(false), "{r:?}");
    assert_eq!(sc(&r)["credits"], 40);
    assert_eq!(sc(&r)["runs"], "cloud");
    assert_eq!(sc(&r)["files"][0]["seed"], Value::Null);
    let spent = std::fs::read_to_string(env.ledger()).unwrap();
    assert_eq!(spent.lines().count(), 2, "{spent}");
    let r = client
        .call_tool(call(
            "generate_sfx",
            json!({"prompt": "horn two", "model": "mock-cloud", "variations": 2, "duration_s": 2.0,
                   "output_dir": "game/cloud"}),
        ))
        .await
        .unwrap();
    assert_eq!(r.is_error, Some(true));
    assert_eq!(sc(&r)["error"]["code"], "credit_limit");
    assert!(
        sc(&r)["error"]["message"]
            .as_str()
            .unwrap()
            .contains("daily")
    );
    assert_eq!(
        files(&env.out("cloud")).len(),
        6,
        "only the first call's files exist"
    );
    assert_eq!(
        std::fs::read_to_string(env.ledger())
            .unwrap()
            .lines()
            .count(),
        2
    );

    let status = client
        .call_tool(call("account_status", json!({})))
        .await
        .unwrap();
    let v = sc(&status);
    assert_eq!(v["ledger"]["today"], 40);
    assert_eq!(v["ledger"]["daily_limit"], 60);
    assert_eq!(v["readable"], false);
    assert_eq!(v["credentials_present"], false);
    client.cancel().await.unwrap();
}

#[tokio::test]
async fn list_models_and_unload_work_without_loading_anything() {
    let env = env();
    let client = spawn(TaskClient, &env).await;
    let r = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    let v = sc(&r);
    let model = |id: &str| {
        v["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .unwrap_or_else(|| panic!("{id}"))
            .clone()
    };
    assert_eq!(model("mock-local")["status"], "ready");
    assert_eq!(model("mock-local")["loaded"], false);
    assert_eq!(model("mock-cloud")["runs"], "cloud");
    assert_eq!(
        model("mock-cloud")["privacy"],
        "the prompt is sent to the mock cloud"
    );
    assert_eq!(v["default_model"], "mock-local");
    assert_eq!(v["jobs_busy"], 0);
    let u = client
        .call_tool(call("unload_models", json!({})))
        .await
        .unwrap();
    assert_eq!(sc(&u)["unloaded"], json!([]));
    client.cancel().await.unwrap();
}
