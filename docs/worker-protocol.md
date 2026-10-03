# sa3_mlx_worker protocol 1

Worker: `worker/sa3_mlx_worker.py`. Stand-in for tests: `src/bin/fake_worker.rs` (header lists its `FAKE_WORKER_MODE` values).

## Starting it

```
<sa3 dir>/.venv/bin/python -u <worker.py> <sa3 dir>        # <sa3 dir> = .../stable-audio-3/optimized/mlx
env: HF_HUB_OFFLINE=1  TQDM_DISABLE=1  PYTHONUNBUFFERED=1
```

- stdin: one JSON object per line (requests). stdout: one JSON object per line (events), nothing else, ever. The worker keeps a
  private dup of fd 1 for protocol lines and `dup2(2, 1)`s, so library prints (Python and C) land on **stderr**; keep its tail
  for error reports. Every event has an `"event"` key.
- The server should start it in its own process group (`own_process_group`) and `kill_tree` on cancel. The worker does not set
  a group itself. It sets its title with `setproctitle` only if that is importable; the command line always contains
  `sa3_mlx_worker`.
- Requests are handled one at a time, in order, on the main thread. There is no cancel message: kill the process group
  (a killed job can leave `<id>_<seed>.f32.partial` files in `out_dir`; delete them).
- EOF on stdin: the worker unloads and exits 0. `unload`: it replies `unloaded` and exits 0 (a later job needs a new process).
- Exit codes: 0 normal, 2 usage (no argument), 3 missing weights at startup.
- Never downloads. The three weight files are opened by explicit path: `<dir>/models/mlx/dit_sm-sfx_f16.npz`,
  `t5gemma_f16.npz`, `same_s_decoder_f32.npz` (symlinks into the HF cache are fine; a dangling link counts as missing).

## Worker to server

First line after start is exactly one of:

```json
{"event":"hello","protocol":1,"pid":8730,"python":"3.11.14","mlx":"0.32.2","revision":"da6edc54ddba10bfd79a077102ded687f80e882b","weights_ok":true,"sa3_dir":"/abs/dir"}
{"event":"error","id":"","stage":"startup","kind":"missing_weights","message":"missing weight file(s) under /abs/dir/models/mlx: t5gemma_f16.npz","hint":"..."}   // then exit 3
```

`mlx` is the installed `mlx` distribution version (null if absent); `revision` is the HF snapshot directory name the DiT symlink
points into (null when the file is not a symlink into a `snapshots/<rev>/` path). The server should refuse a `protocol` other than 1.

Per request:

| event | fields |
|---|---|
| `pong` | `id` (echo of the request id, `""` if none) |
| `loading` | `id`. Only on a cold load (first `generate` in this process, or the first after nothing was resident). Takes ~1.2 s. |
| `progress` | `id`, `stage` (`condition` \| `sample` \| `decode`), `step`, `total`, `item` (int or `null`) |
| `result` | see below |
| `error` | `id`, `stage`, `kind`, `message`, `hint` |
| `unloaded` | `id`. Last line before exit 0. |

All fields above are always present; `progress` always has all six keys.

### `progress`

`step` is the number of units already finished, `total` the units in the stage. Stages come in order, once per job:

| stage | events | `step` / `total` | `item` |
|---|---|---|---|
| `condition` | 1, at stage start | `0` / `1` | null |
| `sample` | 1 at start, then one per sampler step | `0..steps` / `steps` (8 by default) | null (the batch steps together) |
| `decode` | one per clip, sent when that clip starts | `0..n-1` / `n` (n = number of seeds) | clip index |

The first event of a stage marks the stage change; there is no explicit "done" event, the next stage's first event or the
`result` closes it. Typical timing on the M4 (warm, 1 s clip): condition 0.04 s, sample 0.7 s (8 steps, ~90 ms each), decode 0.04 s per clip.

### `result`

```json
{"event":"result","id":"j1","sample_rate":44100,"channels":2,"frames":44100,
 "files":["/abs/out_dir/j1_101.f32","/abs/out_dir/j1_102.f32"],"seeds":[101,102],
 "timings":{"load":1.181,"condition":0.132,"sample":1.681,"decode":0.176,"total":3.172},
 "peak_mlx_mb":2485.1,"active_mlx_mb":1621.9,"peak_rss_mb":1757.9}
```

- `files[i]` belongs to `seeds[i]`, in request order. Absolute paths (`out_dir` is made absolute by the worker). Name:
  `<out_dir>/<id>_<seed>.f32`.
- Each file is **raw little-endian float32, interleaved L R L R ...**, `frames * channels * 4` bytes, no header, not clipped
  (peaks can exceed 1.0). Written as `<name>.f32.partial` and renamed by the worker only after every clip passed the NaN/Inf
  check, so on a `result` all files are final and on an `error` none are left behind (a crash or kill can leave `.partial`).
- `frames` = round(seconds * 44100) (the model works in 4096-sample blocks; the tail is cut off). Same for every file.
- `timings` (seconds, 3 decimals): `load` is 0.0 on a warm job. `total` covers the whole request including load and file writing.
- `peak_mlx_mb`: MLX peak allocation during this job (MiB, includes the load on a cold job). `active_mlx_mb`: MLX memory in use
  at the end (models + buffers, ~1.6 GB). `peak_rss_mb`: the process's peak RSS so far (`ru_maxrss`; not reset per job and it
  under-reports Metal memory, use `phys_footprint` / the memory census for the real figure).

## Server to worker

```json
{"op":"ping","id":"p1"}
{"op":"generate","id":"j1","prompt":"...","seconds":1.0,"steps":8,"seeds":[101,102,103,104],"out_dir":"/abs/dir"}
{"op":"unload","id":"u1"}
```

`generate` rules (anything else is `bad_request`, stage `request`, and the worker stays alive):

| field | rule |
|---|---|
| `id` | string, 1-64 chars of `A-Za-z0-9._-` (becomes part of the file names) |
| `prompt` | non-empty string after trimming (T5Gemma truncates at 256 tokens, silently) |
| `seconds` | JSON number (not bool) from 0.1 to 120 |
| `steps` | integer 1 to 50, default 8 |
| `seeds` | list of 1 to 8 distinct integers, each 0 <= s < 2^63 |
| `out_dir` | existing directory (relative paths are resolved against the worker's cwd) |

Unknown fields are ignored. Unparseable lines and unknown ops give `{"event":"error","id":"<id or ''>","stage":"protocol","kind":"bad_request",...}`.

Generation: all seeds run as one batch (same length). Noise per seed is `mx.random.normal(key=mx.random.key(seed))`, as the CLI;
the sampler's own noise key is `seeds[0] + 1`. So **a clip is a function of the whole seed list**: seed 101 in `[101]` differs
from seed 101 in `[101, 102]`; the same list always reproduces exactly (checked across processes). Record batch size and index
in the sidecar. cfg and negative prompts are not exposed (no effect on sm-sfx).

## Errors

`{"event":"error","id","stage","kind","message","hint"}`. `stage` is where it happened: `startup`, `protocol`, `request`,
`loading`, `condition`, `sample`, `decode`, `write`. After any job error the worker stays alive and the next request works.

| kind | when | notes |
|---|---|---|
| `bad_request` | validation, bad JSON, unknown op | caller error; map to an `isError` result, not a server fault |
| `missing_weights` | startup (exit 3), or a weight file vanishing before load (`stage: loading`) | |
| `nan_output` | NaN/Inf in a decoded clip (stage `decode`) | nothing written |
| `out_of_memory` | MLX allocation failure (message contains `[metal::malloc]`, "out of memory", ...), or `MemoryError` | |
| `worker_exception` | anything else | message is `ExcType: text` (first line); the traceback is on stderr |

A worker crash (no `error` line, process gone, usually non-zero exit) is the server's to detect: treat EOF on stdout as a crash and
report the stderr tail.

## Measured (M4, 24 GB, real weights, Unreal Editor open at ~27 GB)

- Cold load 1.2 s (`load`), steady state ~1.6 GB MLX / ~2.35 GB `phys_footprint`.
- 1 s clip x 4 seeds, cold: 3.17 s total; warm 2 s x 2 seeds: 1.4 s; warm 1 s x 1 seed: 0.81 s.
- MLX peak 2.2-2.5 GB for 1 s clips, 2.9 GB for 4 x 30 s. Process `peak memory footprint` (what `/usr/bin/time -l` and the memory
  census see) is higher: 3.66 GB after the first cold run (np.load copies during load), 4.5 GB after 4 x 30 s. Budget the
  census/headroom check with ~3.7 GB, not the 2.7 GB in the design (`memory_mb`).
