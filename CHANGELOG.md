# Changelog

## 0.1.0 (2026-10-03)

First version: text-to-sound-effect over MCP, with a local and a cloud backend.

- Tools: `generate_sfx` (MCP Task on 2026-07-28 clients, `wait: false` + `get_job` for others), `get_job`, `cancel_job`,
  `list_models`, `account_status`, `unload_models`. Every tool has a typed output schema. Mistakes in a call come back as
  `isError` results with a code and a fix.
- `sa3_mlx` backend: Stable Audio 3 small-sfx on Apple Silicon through MLX. A Python worker embedded in the server runs
  inside the stable-audio-3 venv over JSON lines (protocol 1), with per-step progress, batching by seeds, a NaN guard, cancel
  by process-group kill, idle unload and explained failures (`nan_output`, `out_of_memory`, `missing_weights`,
  `worker_exception`, a crash with the log tail). Weights load by explicit path, offline.
- `elevenlabs` backend: Sound Effects over HTTPS with a concurrency limit, retries on busy responses, credits read from
  `character-cost`, and `account_status` that works with scoped keys lacking `user_read`.
- Cost control: estimate before sending, a per-call and a daily credit limit, and a JSON-lines ledger. A refused call sends
  nothing.
- Secrets: the key comes from `ELEVENLABS_API_KEY`, else the macOS Keychain (the `security` CLI) or the OS keyring elsewhere.
  It is never written to config, sidecars, logs or results, and error messages are scrubbed of `sk_...`.
- Audio pipeline in Rust: loop (crossfade for local models), trim, mono, 48 kHz resampling, loudness, a true-peak limiter
  (-1 dBTP by default), fades, 16- or 24-bit output. Each clip also keeps an untouched raw copy and a sidecar with
  parameters, measurements, waveform peaks, licences and warnings.
- Short sounds: Stable Audio 3 small smears clips under about 1.5 s, so the local backend generates anything shorter at
  `options.min_generate_s` (default 2 s) and cuts each clip back to its event (onset to decay, at most the requested length,
  with short fades). `trim_silence` also treats anything 45 dB under the clip's peak as silence.
- Model registry (`models.toml`) with five statuses (`ready`, `missing_files` with the exact `hf download` command,
  `needs_licence_acceptance`, `needs_credentials`, `backend_unavailable`) and every component's licence, commercial use and
  privacy.
- Machine safety: two lanes (local and cloud), a cross-process GPU lock shared with sister servers, a free-memory preflight
  that names heavy processes (including the Python worker), and idle unload.
- Output: files jailed to allowed roots, `*.partial` renamed on success and deleted on cancel or failure, never
  overwriting.
- Known: `serverInfo` reports `rmcp` until `adk-mcp-sdk`'s `mcp_2026_server!` takes a name. Deferred: the 2025-11-25 Tasks
  design and an MCP App review view.
