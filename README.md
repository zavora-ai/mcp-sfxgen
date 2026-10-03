# Sound Effects MCP Server (mcp-sfxgen)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94.1%2B-orange.svg)](https://www.rust-lang.org/)
[![MCP](https://img.shields.io/badge/MCP-2025--11--25%20%7C%202026--07--28-green.svg)](https://modelcontextprotocol.io/)
[![ADK-Rust Enterprise](https://img.shields.io/badge/ADK--Rust-Enterprise-purple.svg)](https://enterprise.adk-rust.com)
[![Registry Ready](https://img.shields.io/badge/ADK_Registry-Ready-green.svg)](https://www.zavora.ai)

**Text-to-sound-effect** over MCP. A prompt goes in; mastered WAV files come out, each with an untouched raw copy and a
provenance sidecar. It was built for agents making game audio on a workstation shared with Blender and Unreal Engine: one
local job at a time, a memory check before loading, explained failures, a hard cap on cloud spending and no surprises on disk.

Two backends sit behind one interface:

- **Local:** [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) small-sfx on Apple Silicon through MLX. It
  is free and private, and it takes about 1 second per clip once the model is loaded.
- **Cloud:** [ElevenLabs Sound Effects](https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert). It costs
  credits and sends the prompt to ElevenLabs, so it is never chosen for you.

The server itself builds and runs on **macOS, Linux and Windows**. A backend that can't run on the current machine says why
in `list_models`.

## Example

A call to `generate_sfx`:

```json
{
  "prompt": "A thin, feeble, short car horn beep from a small, tired old compact car. One quick meep. Dry, close, no music.",
  "duration_s": 1.0,
  "variations": 4,
  "name": "meep"
}
```

With the local model this writes, into the configured output folder:

```
meep_01.wav  meep_01.raw.wav  meep_01.json
meep_02.wav  meep_02.raw.wav  meep_02.json
meep_03.wav  meep_03.raw.wav  meep_03.json
meep_04.wav  meep_04.raw.wav  meep_04.json
```

The result lists each file with its duration, loudness, true peak and seed, and carries one `resource_link` per file.
On an M4 (24 GB) the four 1-second clips took about 3 seconds from a cold start, with the model load included.

## Architecture

- **Backends** implement one trait (`availability`, `generate` with stage progress and cancel, `unload`, `account`). Each sits
  behind a Cargo feature: `sa3_mlx` drives a small Python worker inside the Stable Audio 3 venv over JSON lines
  ([`docs/worker-protocol.md`](docs/worker-protocol.md)), `elevenlabs` talks HTTPS, and `mock` is for tests. The worker file is
  embedded in the server and written to the cache dir, so the install stays one binary and the Stable Audio 3 checkout stays
  unmodified.
- **Model registry** (`models.toml`): adding a model on an existing backend is config only. Files are checked in the Hugging
  Face cache, in the pinned revision. The server **never downloads**: `list_models` prints the exact `hf download` command and
  size for anything missing.
- **Job manager:** two lanes, local and cloud, each with a bounded queue. A cloud call never waits behind a local one. The local
  lane takes a cross-process GPU lock for each job and runs a free-memory check before the model loads.
- **Audio pipeline:** every clip goes through loop, edits, a true-peak limiter and a measured write, in Rust. See
  [Outputs](#outputs).
- **Safety:** output paths are jailed to configured roots (symlink and `..` escapes are rejected). Files are written as
  `*.partial` and renamed only on success. Cancel and failure delete them. Nothing is ever overwritten. The worker runs in its
  own process group and is killed as a tree on cancel or shutdown.

## Tools (6)

| Tool | Purpose | Risk |
|------|---------|------|
| `generate_sfx` | Text to up to 4 WAV variations, each with a raw copy and a sidecar. MCP Task on 2026-07-28 clients | internal_write |
| `get_job` | Status, stage progress and result of a job started with `wait: false` | read_only |
| `cancel_job` | Cancel a queued or running job. A local worker is stopped and no partial files are left | internal_write |
| `list_models` | Status (`ready`, `missing_files`, `needs_licence_acceptance`, `needs_credentials`, `backend_unavailable`, each with the fix), licences, commercial use, privacy, memory needed, typical time, duration range. Never loads weights or spends credits | read_only |
| `account_status` | ElevenLabs plan and credits when the key can read the account, plus the local credit ledger totals and limits. Free and cached | read_only |
| `unload_models` | Stop the kept-alive local worker to free its memory. Refused while local jobs are queued or running | internal_write |

`generate_sfx` parameters:

| Parameter | Meaning |
|---|---|
| `prompt` (required) | What the sound is, in plain words: the object, the action, the character. At most 4000 characters (Stable Audio 3 reads about 256 tokens) |
| `model` | A model id from `list_models`. Default: the configured `default_model` |
| `duration_s` | Length in seconds within the model's range. Omitted: local models make 2 s; ElevenLabs lets the model pick the length (a flat, higher price) |
| `variations` | 1 to 4 (default 1), written as `<name>_01`, `<name>_02`, ... |
| `seed` | Local only. Variation n uses seed + n - 1. Omitted: a random seed, always reported. Ignored by ElevenLabs, with a warning |
| `loop` | A seamless loop of exactly `duration_s`. Not with `post.trim_silence` or fades |
| `name` | File name stem: letters, digits, `_` and `-`, at most 64. Default: a slug of the prompt |
| `output_dir` / `output_path` | A folder, or one exact `.wav` path (one variation only), inside an allowed root. Not both, and `output_path` doesn't combine with `name` |
| `prompt_influence` | Cloud only, 0 to 1. Ignored by local models, with a warning |
| `steps` | Local only, 1 to 50 (default 8). Ignored by ElevenLabs, with a warning |
| `post` | Post-processing, below |
| `wait` | Default true. `false` returns a `job_id` at once |

`post` options: `trim_silence`, `fade_in_ms`, `fade_out_ms` (0 to 5000), `loudness_lufs` (-70 to 0), `true_peak_dbtp` (-20 to
0, or `null` to turn the limiter off), `mono`, `sample_rate` (44100 or 48000) and `bit_depth` (16 or 24).

Long jobs: `generate_sfx` runs as a protocol-native **MCP Task** for clients that support them, blocks for legacy clients
(Claude Code backgrounds calls over two minutes automatically), and accepts `wait: false`, which returns a `job_id` to poll
with `get_job` and stop with `cancel_job`.

Mistakes in a call (a bad parameter, a duration out of range, an option conflict, a path outside the roots, a model that
isn't ready, a missing key, a credit limit) come back as `isError` results with `{error: {code, message, fix}}`, so the
model can correct itself. Everything else is a JSON-RPC error with `data.code`. Each tool has a typed output schema.

## Install

```bash
git clone https://github.com/zavora-ai/mcp-sfxgen && cd mcp-sfxgen
cargo install --path .
```

The binary builds on any platform. The local model needs the steps below; the ElevenLabs backend needs only a key (see
[Configure](#configure)).

### Stable Audio 3 on MLX (Apple Silicon)

```bash
git clone https://github.com/Stability-AI/stable-audio-3 && cd stable-audio-3/optimized/mlx
./install.sh
```

`install.sh` creates a `.venv` there (Python 3.11 through `uv`; about 266 MB) and offers to download weights. The server
runs `<dir>/.venv/bin/python`, with `<dir>` as the `optimized/mlx` folder, and loads the weights by explicit path from
`<dir>/models/mlx`. Choose the `sm-sfx` DiT in the installer so those links are made. If you fetch the weights yourself, link
the three files into `<dir>/models/mlx`.

The weights are about 1.6 GB, pinned to one revision. Fetch them with the
[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`pip install -U huggingface_hub`). The
repository isn't gated, but it carries the Stability AI Community License (see [Licences](#licences)).

```bash
hf download stabilityai/stable-audio-3-optimized MLX/dit_sm-sfx_f16.npz MLX/t5gemma_f16.npz MLX/same_s_decoder_f32.npz --revision da6edc54ddba10bfd79a077102ded687f80e882b
```

`list_models` always shows which files are still missing, with the command for them (it adds `--cache-dir` when your cache
isn't the default one). The local model also stays at `needs_licence_acceptance` until you set `licence_accepted` in the
config.

## Configure

Claude Code (`.mcp.json`):

```json
{
  "mcpServers": {
    "sfxgen": {
      "type": "stdio",
      "command": "/Users/you/.cargo/bin/mcp-sfxgen",
      "env": {
        "MCP_SFXGEN_CONFIG_DIR": "/path/to/project/.sfxgen",
        "ELEVENLABS_API_KEY": "${ELEVENLABS_API_KEY:-}"
      }
    }
  }
}
```

On first run the server writes `config.toml` and `models.toml` to `$MCP_SFXGEN_CONFIG_DIR`, or by default to the platform
config dir (`~/Library/Application Support/mcp-sfxgen` on macOS, `$XDG_CONFIG_HOME/mcp-sfxgen` or `~/.config/mcp-sfxgen` on
Linux, `%APPDATA%\mcp-sfxgen` on Windows). Relative paths in the config resolve against the config dir's parent, so a project
keeps its config in `<project>/.sfxgen/` and lists roots like `"assets"`. See [`config.example.toml`](config.example.toml)
and [`models.example.toml`](models.example.toml). Walk-through of `config.toml`:

```toml
allowed_roots = ["generated", "my-game"]     # sounds are written to, and only to, these
default_output_dir = "generated/sfx"         # must be inside a root
default_model = "sa3-small-sfx"              # a local model; ElevenLabs is used only when a call names it
memory_headroom_mb = 1000                    # free memory that must remain after the local model loads

[sa3_mlx]
dir = "~/tools/stable-audio-3/optimized/mlx"
licence_accepted = "2026-10-03"              # the date you accepted the licences below; empty = the model won't run

[elevenlabs]
plan = "free"                                # "free" marks every ElevenLabs file non-commercial
concurrency = 2                              # parallel requests the plan allows
per_call_credit_limit = 300
daily_credit_limit = 2000
```

Other keys: `heavy_processes` (named in memory refusals), `job_ttl_s` (how long finished jobs stay queryable, default
3600), `queue_depth` (jobs waiting per lane, default 4), `idle_unload_s` (default 600; 0 unloads only on `unload_models`),
`hf_cache`, the `[elevenlabs]` rates and `timeout_s`, and `[post] true_peak_dbtp`.
`MCP_SFXGEN_CACHE_DIR` and `MCP_SFXGEN_DATA_DIR` move the cache (it moves the shared GPU lock too) and the credit ledger.

**`licence_accepted` is yours to set.** The server never fills it in. Read the terms first (see [Licences](#licences)).

### The ElevenLabs key

The key is read from the environment variable named by `api_key_env` (default `ELEVENLABS_API_KEY`), else from the macOS
Keychain item named by `keychain_service` (default `elevenlabs-api-key`; on other platforms, the OS keyring). It is never
written to a config file, a sidecar, a log or a result, and every error message is scrubbed of `sk_...` strings.

To use the Keychain, create the item once. `security` prompts for the key, so it never lands in shell history:

```bash
security add-generic-password -a "$USER" -s elevenlabs-api-key -w
```

The server reads it by running `security find-generic-password`, because an item made by the `security` tool only lets that
tool read it without a prompt. A headless server can't answer a prompt.

## Outputs

Each variation writes a family of three files, with a shared stem and a number:

| File | What it is |
|---|---|
| `<name>_NN.wav` | The processed copy: 44.1 kHz, 16-bit (dithered), true peak at or below -1 dBTP by default |
| `<name>_NN.raw.wav` | The backend's output, converted to WAV and otherwise untouched: 32-bit float for local models, 16-bit PCM as delivered for ElevenLabs |
| `<name>_NN.json` | The sidecar |

A stem is free only when no file in the folder belongs to its family, and all variations are reserved together, so two
jobs never share a name. With `output_path` the names are exact (`<stem>.wav`, `<stem>.raw.wav`, `<stem>.json`), and an
existing family is an `output_exists` error.

The processed copy is made in this order: loop (local models: generate extra seconds, drop the model's fade-out tail and
crossfade the end into the head; ElevenLabs loops natively and the clip may run slightly long), trim silence, mono
downmix, resample to 48 kHz, loudness normalisation, the true-peak limiter, fades, then quantising to 16 or 24 bits. Only the
limiter is on by default. It uses 4x oversampled peak detection, 1.5 ms look-ahead and a 50 ms release, shares one gain
across channels, and the result is verified with a final true-peak measurement. A limiter that works hard (over 2 dB) adds a
warning.

Numbers in the result and sidecar are measured on the WAV as written: duration, integrated loudness (`null` under 0.4 s),
true peak, limiter gain, and for loops the seam (the jump at the wrap against the signal's typical step; about 1 is
seamless).

The sidecar holds the generator version, job id, file names, model, backend, where it ran, provider, the prompt, the seed
and batch size and index, the resolved parameters, the steps applied, measurements, 512 waveform min/max pairs, timings, the
credits charged, the plan, `commercial_use` and its note, every licence and notice, and warnings.

A local clip is a function of the whole seed list: seed 101 in a batch of 4 differs from seed 101 on its own. The same
list reproduces the same clips, so the sidecar records the batch.

## Costs

Local generation is free. ElevenLabs charges credits, so every cloud call is checked before anything is sent.

- **Rate:** 10 credits per second of audio when `duration_s` is given, 50 when the model picks the length. These are
  measured on a Free plan (2026-10-03) and set in `[elevenlabs]` (`credits_per_second`, `credits_auto_duration`). The real
  charge is read from each response's `character-cost` header.
- **Limits:** a call that would cost more than `per_call_credit_limit` (default 300), or that would take the day past
  `daily_credit_limit` (default 2000, summed by local calendar day), is refused with a `credit_limit` error and sends nothing.
  The estimate is credits per second times the duration, rounded up per clip, times the number of variations.
- **Ledger:** every charged call is appended to `<data dir>/mcp-sfxgen/credits.jsonl` (JSON lines: time, model, credits,
  duration, file). Credits spent on clips that were thrown away, because a sibling failed or the job was cancelled, are
  still recorded.
- **`account_status`** shows the ledger's totals for today and this month, the limits and the rates. It adds the ElevenLabs
  plan and credits left when the key can read the account. A scoped key without the `user_read` permission can't: the tool
  then returns `readable: false` with the reason, and the ledger still works. If the account is readable, its plan
  overrides `plan` in the config.

## Licences

`list_models` shows these per model, and each sidecar records them. Check the terms before shipping outputs.

| Component | Licence | Commercial use |
|---|---|---|
| [Stable Audio 3 small-sfx weights](https://huggingface.co/stabilityai/stable-audio-3-optimized) | [Stability AI Community License](https://stability.ai/community-license-agreement) | conditional: free under US$1M annual revenue after registering with Stability |
| T5Gemma text encoder | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | conditional |
| [stable-audio-3](https://github.com/Stability-AI/stable-audio-3) code | MIT | yes |
| ElevenLabs output | ElevenLabs Terms of Use | conditional: paid plans only. The Free plan is non-commercial |
| mcp-sfxgen | Apache-2.0 | yes |

The Community License also asks for a "Powered by Stability AI" notice, and outputs may not be used to train a foundation
model. With `plan = "free"` the server reports every ElevenLabs file as `commercial_use: no`, with the reason in the
sidecar's notices. ElevenLabs may use prompts to improve its service unless you opt out in the account's data-use settings.

## Models

| id | Backend | Runs | Duration | Memory | Typical time | Commercial use | Privacy |
|---|---|---|---|---|---|---|---|
| `sa3-small-sfx` | `sa3_mlx` | local | 0.5 to 30 s | 3.7 GB process footprint, plus headroom | about 1 s per clip warm (`typical_s` 1.5) | conditional | stays on the machine |
| `elevenlabs-sfx-v2` | `elevenlabs` | cloud | 0.5 to 30 s | none | `typical_s` 4 | conditional (`no` on the Free plan) | the prompt is sent to ElevenLabs |

Statuses: `ready`; `missing_files` (with the exact `hf download` command and size); `needs_licence_acceptance` (what to
accept); `needs_credentials` (where to put the key); `backend_unavailable` (why, for example not macOS arm64 or no venv).
`elevenlabs-sfx-v2` has a native loop. `sa3-small-sfx` loops by crossfade. The model allows 120 s, but 30 s is plenty for
effects and the limit is set in `models.toml`.

## Machine safety

Stable Audio 3 shares a 24 GB machine with Unreal Editor and Blender, and with the image and mesh servers.

- **GPU lock:** each local job takes the cross-process lock `zavora-mcp/local-gpu.lock` (holder `mcp-sfxgen`), shared with
  sister servers such as mcp-meshgen. The lock is held per job, not while the worker sits idle, so a mesh job isn't blocked
  for ten minutes.
- **Memory preflight:** before loading, the server checks that `memory_mb + memory_headroom_mb` is free (3700 + 1000 MB by
  default) and refuses rather than swapping. It names the heavy processes it sees, for example *"close UnrealEditor"*. On
  macOS it reads each process's physical footprint, which includes GPU memory. The census also names the Python worker by its
  command line.
- **Measured:** the worker's process footprint peaked at about 3.7 GB after the first cold run. After four 30-second clips it
  reached 4.5 GB.
- **Idle unload:** the worker stays loaded for `idle_unload_s` (default 600 s) so the next call skips the load, then exits.
  `unload_models` stops it now. A cancel or crash kills the worker's process group.
- The cloud lane takes no lock and runs no preflight.

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features        # unit tests, the mock-backend protocol tests (MCP 2025-11-25 and 2026-07-28), the
                                 # sa3_mlx backend against a fake worker, the ElevenLabs backend against a fake HTTP server
```

Features: `sa3_mlx` and `elevenlabs` (both default) and `mock` (a deterministic fake backend, the `fake-worker` binary, and
`MCP_SFXGEN_MOCK_DELAY_MS` / `MCP_SFXGEN_MOCK_FAIL` for tests). CI also runs clippy without default features and builds on the MSRV.

**Live tests** are `#[ignore]`d and run by hand:

```bash
cargo test --release --test live -- --ignored --nocapture
```

| Test | Needs | Cost |
|---|---|---|
| `live_sa3` | Apple Silicon, the MLX venv and weights (`MCP_SFXGEN_SA3_DIR` overrides the checkout path) | free; about 3.7 GB of memory |
| `live_elevenlabs` | `ELEVENLABS_API_KEY` | **10 credits** (one 1-second clip) |

A test checks that the `hf download` command in this README matches what `list_models` prints for an empty cache.

## rmcp and MCP compatibility

This server is built with [`rmcp` 3.1.2](https://github.com/modelcontextprotocol/rust-sdk/releases/tag/rmcp-v3.1.2) and requires
Rust 1.94.1 or newer. It keeps legacy MCP initialization compatibility and targets MCP protocol revisions `2025-11-25` and
`2026-07-28`. Every tool works with a plain 2025-11-25 client, and the protocol tests run one of those and one 2026-07-28
Tasks client.

## MCP 2026-07-28 rollout

This server uses `rmcp` 3.1.2 and `adk-mcp-sdk` 0.3 (pinned to `fd08c98` for `task_ttl_overrides` and task status updates).
It accepts stateless MCP 2026 requests with per-request protocol, client identity and capability metadata, and keeps the
legacy MCP 2025-11-25 initialize flow.

- **Tasks:** `generate_sfx` (the Tasks extension through `adk-mcp-sdk`'s `mcp_2026_server!`; TTL 1 hour). Clients without the
  extension block, or pass `wait: false` and poll `get_job`.
- **MRTR approvals:** none. Writes are confined to configured output roots (`writes_allowed = "gated"`), and the cost guard
  refuses over-limit cloud calls before they are sent.
- **Annotations:** `generate_sfx`, `cancel_job` and `unload_models` are mutating, and `cancel_job` and `unload_models` are
  idempotent. `generate_sfx` and `account_status` are open-world (they can reach ElevenLabs).
- **Caching:** `tools/list` returns a public `ttlMs` of 60,000 for MCP 2026 clients.
- **Transport:** stdio. Task records are process-local.
- **Deferred:** the 2025-11-25 Tasks design, because rmcp 3.x dropped it on purpose and it needs a translation layer in the
  SDK; and an MCP App review view (listen, keep, discard, regenerate), because it doesn't block agents. Today you open the WAV
  files directly.

## Known limitations

- `serverInfo` reports `rmcp` and its version rather than `mcp-sfxgen`. The `mcp_2026_server!` macro hard-codes the name;
  fixing it needs a change in `adk-mcp-sdk`.
- The MLX backend runs on macOS with Apple Silicon only. Elsewhere `sa3-small-sfx` reports `backend_unavailable`, and
  ElevenLabs still works.
- ElevenLabs has no seed, so its results can't be reproduced. A `seed` there is ignored with a warning.
- A local batch isn't bit-identical to the same seed on its own (see [Outputs](#outputs)).
- Prompts past about 256 tokens are cut silently by the local text encoder.
- Audio-to-audio and inpainting aren't exposed.

## License

Apache-2.0 for this server. Models and backends carry their own licences (see [Licences](#licences)); check them before
shipping outputs.
