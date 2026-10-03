"""mcp-sfxgen worker for Stable Audio 3 (small-sfx) on Apple Silicon, via MLX.

Runs inside the stable-audio-3 MLX venv:  <venv python> -u sa3_mlx_worker.py <stable-audio-3/optimized/mlx dir>
The server embeds this file and writes it to its cache dir, so the checkout stays unmodified.
Environment from the server: HF_HUB_OFFLINE=1, TQDM_DISABLE=1, PYTHONUNBUFFERED=1. Nothing is ever downloaded: the three weight
files are opened by explicit path under <dir>/models/mlx/, and a missing one is reported as `missing_weights`.

Protocol 1: one JSON object per line. Protocol lines go to a private duplicate of the real stdout; everything the libraries
print (Python or C level) goes to stderr, and the server keeps its tail for error reports. Full schemas: PROTOCOL.md.
  server -> worker  {"op":"ping","id"}  {"op":"generate","id","prompt","seconds","steps":8,"seeds":[...],"out_dir"}
                    {"op":"unload","id"}   (EOF also ends the worker)
  worker -> server  hello, pong, loading, progress, result, error, unloaded

All MLX work runs on the main thread. Requests are handled one at a time, in order.
"""
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

PROTOCOL = 1

# Keep a private handle on the real stdout for protocol lines, then point fd 1 (C code included) and sys.stdout at stderr.
_proto = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLES_PER_LATENT = 4096  # PatchedPretransform downsample x SAME-S 16x expansion
WEIGHTS_SUBDIR = Path("models") / "mlx"
DIT_FILE = "dit_sm-sfx_f16.npz"
T5_FILE = "t5gemma_f16.npz"
DECODER_FILE = "same_s_decoder_f32.npz"
WEIGHT_FILES = (DIT_FILE, T5_FILE, DECODER_FILE)
DECODER_CHUNK, DECODER_OVERLAP = 8, 2  # same-s, as DECODER_CHOICES in scripts/sa3_mlx.py

MIN_SECONDS, MAX_SECONDS = 0.1, 120.0
MAX_STEPS = 50
MAX_SEEDS = 8
MAX_SEED = 2**63  # exclusive
ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # ids become part of file names

OOM_MARKERS = ("out of memory", "insufficient memory", "[metal::malloc]", "failed to allocate", "maximum allowed buffer size")
OOM_HINT = "close other heavy apps (Unreal, Blender, other models) and retry; fewer seeds or a shorter clip also helps"


def emit(event, **fields):
    try:
        _proto.write(json.dumps({"event": event, **fields}) + "\n")
        _proto.flush()
    except BrokenPipeError:
        os._exit(0)  # the server is gone; nobody is listening


class WorkerError(Exception):
    def __init__(self, stage, kind, message, hint=""):
        super().__init__(message)
        self.stage, self.kind, self.message, self.hint = stage, kind, message, hint


def error_fields(e):
    return {"stage": e.stage, "kind": e.kind, "message": e.message, "hint": e.hint}


# ---------------------------------------------------------------------------------------------------------------------
# Weights (checked at startup, never downloaded)

def check_weights(sa3_dir):
    """Returns (missing file names, snapshot revision). A symlink into the Hugging Face cache must resolve to a real file.
    The revision is the snapshot directory name the DiT link points into (.../snapshots/<revision>/MLX/<file>)."""
    folder = Path(sa3_dir) / WEIGHTS_SUBDIR
    missing = [name for name in WEIGHT_FILES if not (folder / name).is_file()]
    revision = None
    dit = folder / DIT_FILE
    try:
        if dit.is_symlink():
            target = Path(os.readlink(dit))
            if not target.is_absolute():
                target = (folder / target)
            parts = target.parts
            if "snapshots" in parts:
                revision = parts[parts.index("snapshots") + 1]
    except (OSError, IndexError):
        pass
    return missing, revision


# ---------------------------------------------------------------------------------------------------------------------
# Request validation

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def validate_generate(req):
    """Returns a clean params dict, or raises WorkerError(stage="request", kind="bad_request")."""

    def bad(message, hint="fix the request; this is a caller error, not a worker fault"):
        raise WorkerError("request", "bad_request", message, hint)

    job_id = req.get("id")
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        bad("id must be a string of 1-64 characters from A-Z a-z 0-9 . _ -", "the id becomes part of the output file names")
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        bad("prompt must be a non-empty string")
    seconds = req.get("seconds")
    if not _is_number(seconds) or not math.isfinite(seconds) or not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        bad(f"seconds must be a number from {MIN_SECONDS} to {MAX_SECONDS:g}, got {seconds!r}")
    steps = req.get("steps", 8)
    if not _is_int(steps) or not (1 <= steps <= MAX_STEPS):
        bad(f"steps must be an integer from 1 to {MAX_STEPS}, got {steps!r}")
    seeds = req.get("seeds")
    if not isinstance(seeds, list) or not (1 <= len(seeds) <= MAX_SEEDS):
        bad(f"seeds must be a list of 1 to {MAX_SEEDS} integers")
    for s in seeds:
        if not _is_int(s) or not (0 <= s < MAX_SEED):
            bad(f"every seed must be an integer from 0 to 2^63-1, got {s!r}")
    if len(set(seeds)) != len(seeds):
        bad("seeds must be distinct", "the output file names are <id>_<seed>.f32")
    out_dir = req.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir or not os.path.isdir(out_dir):
        bad(f"out_dir must be an existing directory, got {out_dir!r}", "the server creates the directory before the request")
    return {
        "id": job_id, "prompt": prompt.strip(), "seconds": float(seconds), "steps": steps, "seeds": list(seeds),
        "out_dir": os.path.abspath(out_dir),
    }


def classify_exception(stage, e):
    """Maps an exception raised while working into the protocol's error kinds."""
    text = f"{type(e).__name__}: {e}"
    lowered = text.lower()
    if isinstance(e, MemoryError) or any(m in lowered for m in OOM_MARKERS):
        return WorkerError(stage, "out_of_memory", text.splitlines()[0][:400], OOM_HINT)
    if isinstance(e, FileNotFoundError) and stage == "loading":
        return WorkerError(stage, "missing_weights", text.splitlines()[0][:400],
                           "a weight file disappeared after startup; restart the worker (list_models shows the download command)")
    return None


# ---------------------------------------------------------------------------------------------------------------------
# The model (real MLX implementation). Tests replace `Engine` with a stand-in.

class AudioItem:
    """One generated clip as float32 (channels, frames) in a numpy array."""

    def __init__(self, array):
        self.array = array

    def all_finite(self):
        import numpy as np

        return bool(np.isfinite(self.array).all())

    def to_f32le_interleaved(self):
        import numpy as np

        return np.ascontiguousarray(self.array.T, dtype="<f4").tobytes()


class Engine:
    """Stable Audio 3 small-sfx, resident between requests: T5Gemma, conditioner, DiT, SAME-S decoder."""

    def __init__(self, sa3_dir):
        self.dir = Path(sa3_dir)
        self.loaded = False
        self.mx = None  # mlx.core, imported on first load so ping/validation never pay for it

    # -- memory counters -------------------------------------------------------------------------------------------
    def _mem(self, name):
        mx = self.mx
        if mx is None:
            return 0
        fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
        return int(fn()) if fn else 0

    def reset_peak(self):
        mx = self.mx
        if mx is None:
            return
        fn = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
        if fn:
            fn()

    def peak_mb(self):
        return round(self._mem("get_peak_memory") / 1048576, 1)

    def active_mb(self):
        return round(self._mem("get_active_memory") / 1048576, 1)

    def clear_cache(self):
        gc.collect()
        mx = self.mx
        if mx is None:
            return
        fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if fn:
            fn()

    # -- load / unload ---------------------------------------------------------------------------------------------
    def load(self):
        import mlx.core as mx

        self.mx = mx
        sys.path.insert(0, str(self.dir))  # models.defs.* (we never import scripts/, whose wrappers can download)
        from models.defs import same_s_decoder
        from models.defs import sa3_pipeline as p
        from models.defs.dit_mlx import load_dit
        from models.defs.t5gemma_mlx import T5Gemma

        folder = self.dir / WEIGHTS_SUBDIR
        self.p = p
        self.decoder_mod = same_s_decoder
        self.t5 = T5Gemma.from_npz(str(folder / T5_FILE))
        self.padding_emb, self.secs_embedder = p.load_conditioner_from_npz(str(folder / DIT_FILE), prefix="cond.")
        # One loaded DiT serves any length: with no local_add_cond it builds its zero conditioning at the input length.
        self.dit = load_dit(str(folder / DIT_FILE), T_lat=320, dtype=mx.float16, compile_=False)
        self.decoder = same_s_decoder.load_model(weights_path=str(folder / DECODER_FILE), dtype=mx.float32, compile_=False)
        self.loaded = True

    def unload(self):
        for name in ("t5", "padding_emb", "secs_embedder", "dit", "decoder", "decoder_mod", "p"):
            if hasattr(self, name):
                delattr(self, name)
        was_loaded, self.loaded = self.loaded, False
        if was_loaded:
            self.clear_cache()
        else:
            gc.collect()

    # -- one generation --------------------------------------------------------------------------------------------
    def generate(self, prompt, seconds, steps, seeds, progress, timings):
        """Text-to-audio for len(seeds) clips of the same length in one batch. Returns [AudioItem] (one per seed).
        `progress(stage, step, total, item)` is called for each stage event. Fills timings[condition|sample|decode] in s."""
        import numpy as np

        mx, p = self.mx, self.p
        dtype = mx.float16
        batch = len(seeds)
        t_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))

        # 1+2. Text encoder and conditioning (scripts/sa3_mlx.py main(), text-to-audio, cfg 1.0)
        progress("condition", 0, 1, None)
        t0 = time.time()
        embeds, mask = self.t5.encode([prompt], max_len=256)
        mx.eval(embeds, mask)
        embeds_padded = p.apply_prompt_padding(embeds.astype(dtype), mask, self.padding_emb.astype(dtype))
        seconds_embed = self.secs_embedder(seconds).astype(dtype)  # (1, 1, 768)
        cross_attn = mx.concatenate([embeds_padded, seconds_embed], axis=1)  # (1, 257, 768)
        global_cond = seconds_embed[:, 0, :]  # (1, 768)
        if batch > 1:
            cross_attn = mx.concatenate([cross_attn] * batch, axis=0)
            global_cond = mx.concatenate([global_cond] * batch, axis=0)
        mx.eval(cross_attn, global_cond)
        timings["condition"] = time.time() - t0

        # 3. DiT, pingpong sampler. Noise per seed from mx.random.key(seed), as the CLI; sampler key seed+1 (first seed).
        t0 = time.time()
        progress("sample", 0, steps, None)
        sigmas = p.build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
        noise = mx.concatenate(
            [mx.random.normal((1, 256, t_lat), dtype=dtype, key=mx.random.key(s)) for s in seeds], axis=0
        )
        mx.eval(noise)

        def model_fn(x, t):
            return self.dit(x, t, cross_attn, global_cond, local_add_cond=None)

        latents = p.sample_flow_pingpong(
            model_fn, noise, sigmas, seed=seeds[0] + 1,
            on_step=lambda i, total: progress("sample", i, total, None),
        )
        mx.eval(latents)
        timings["sample"] = time.time() - t0

        # 4+5. Decode each item (fp32 SAME-S decoder, chunk dispatch as the CLI), unpatch, trim to the requested length
        t0 = time.time()
        items = []
        requested = int(round(seconds * SAMPLE_RATE))
        for i in range(batch):
            progress("decode", i, batch, i)
            lat = latents[i:i + 1].astype(mx.float32)
            kernel = DECODER_CHUNK + 2 * DECODER_OVERLAP
            if t_lat > kernel:
                patches = self.decoder_mod.decode_chunked(self.decoder, lat, DECODER_CHUNK, DECODER_OVERLAP)
            elif t_lat % 2 == 0:
                patches = self.decoder(lat)
            elif t_lat > 6:
                patches = self.decoder_mod.decode_chunked(self.decoder, lat, 2, 2)
            else:
                lat_even = mx.concatenate([lat, lat[..., -1:]], axis=-1)
                patches = self.decoder(lat_even)[..., : t_lat * 16]
            mx.eval(patches)
            audio = p.patched_decode(patches, patch_size=256, channels=CHANNELS)
            mx.eval(audio)
            audio_np = np.array(audio.astype(mx.float32))[0]  # (2, T)
            if audio_np.shape[-1] > requested:
                audio_np = audio_np[..., :requested]
            items.append(AudioItem(audio_np))
        timings["decode"] = time.time() - t0
        return items


# ---------------------------------------------------------------------------------------------------------------------
# One request

def peak_rss_mb():
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024), 1) if sys.platform == "darwin" else round(peak / 1024, 1)


class Job:
    """Per-request state: current stage (for error reports) and timings."""

    def __init__(self, job_id):
        self.id = job_id
        self.stage = "request"
        self.timings = {}

    def progress(self, stage, step, total, item):
        self.stage = stage
        emit("progress", id=self.id, stage=stage, step=step, total=total, item=item)


def handle_generate(engine, req):
    t_start = time.time()
    raw_id = req.get("id")
    job = Job(raw_id if isinstance(raw_id, str) else "")
    written = []
    try:
        params = validate_generate(req)
        job.id = params["id"]
        job.timings = {"load": 0.0, "condition": 0.0, "sample": 0.0, "decode": 0.0}
        if not engine.loaded:
            job.stage = "loading"
            emit("loading", id=job.id)
            t0 = time.time()
            engine.load()
            job.timings["load"] = time.time() - t0
        engine.reset_peak()
        items = engine.generate(
            params["prompt"], params["seconds"], params["steps"], params["seeds"], job.progress, job.timings
        )
        # Nothing is written until every item has passed the NaN/Inf check.
        job.stage = "write"
        for seed, item in zip(params["seeds"], items):
            if not item.all_finite():
                raise WorkerError(
                    "decode", "nan_output", f"the clip for seed {seed} contains NaN or Inf samples",
                    "retry with another seed or more steps; if it repeats, the model or the GPU is misbehaving (see the worker log)",
                )
        frames = None
        for seed, item in zip(params["seeds"], items):
            data = item.to_f32le_interleaved()
            frames = len(data) // (4 * CHANNELS)
            final = os.path.join(params["out_dir"], f"{job.id}_{seed}.f32")
            partial = final + ".partial"
            written.append(partial)
            with open(partial, "wb") as f:
                f.write(data)
            os.replace(partial, final)
            written[-1] = final
        job.timings["total"] = time.time() - t_start
        emit(
            "result", id=job.id, sample_rate=SAMPLE_RATE, channels=CHANNELS, frames=frames, files=list(written),
            seeds=params["seeds"], timings={k: round(v, 3) for k, v in job.timings.items()},
            peak_mlx_mb=engine.peak_mb(), active_mlx_mb=engine.active_mb(), peak_rss_mb=peak_rss_mb(),
        )
        written = []
    except WorkerError as e:
        emit("error", id=job.id, **error_fields(e))
    except Exception as e:  # MLX and I/O errors surface as various types
        traceback.print_exc(file=sys.stderr)
        mapped = classify_exception(job.stage, e)
        if mapped is not None:
            emit("error", id=job.id, **error_fields(mapped))
        else:
            last = f"{type(e).__name__}: {e}".splitlines()[0][:400]
            emit("error", id=job.id, stage=job.stage, kind="worker_exception", message=last,
                 hint="see the worker log tail below; report it if it repeats")
    finally:
        for path in written:  # a failed job leaves no output files
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            engine.clear_cache()
        except Exception:
            pass


def set_process_title():
    try:
        import setproctitle  # only if it happens to be installed

        setproctitle.setproctitle("sa3_mlx_worker")
    except ImportError:
        pass  # the command line already contains sa3_mlx_worker.py


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        print("usage: sa3_mlx_worker.py <stable-audio-3 optimized/mlx dir>", file=sys.stderr)
        return 2
    sa3_dir = Path(argv[1]).resolve()
    set_process_title()
    missing, revision = check_weights(sa3_dir)
    if missing:
        emit(
            "error", id="", stage="startup", kind="missing_weights",
            message=f"missing weight file(s) under {sa3_dir / WEIGHTS_SUBDIR}: {', '.join(missing)}",
            hint="the worker never downloads; list_models shows the exact download command and size",
        )
        return 3
    import importlib.metadata as md

    try:
        mlx_version = md.version("mlx")
    except md.PackageNotFoundError:
        mlx_version = None
    emit("hello", protocol=PROTOCOL, pid=os.getpid(), python=sys.version.split()[0], mlx=mlx_version,
         revision=revision, weights_ok=True, sa3_dir=str(sa3_dir))

    engine = Engine(sa3_dir)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            emit("error", id="", stage="protocol", kind="bad_request", message=f"not JSON: {e}", hint="")
            continue
        if not isinstance(msg, dict):
            emit("error", id="", stage="protocol", kind="bad_request", message="a request must be a JSON object", hint="")
            continue
        op = msg.get("op")
        msg_id = msg.get("id") if isinstance(msg.get("id"), str) else ""
        if op == "generate":
            handle_generate(engine, msg)
        elif op == "ping":
            emit("pong", id=msg_id)
        elif op == "unload":
            engine.unload()
            emit("unloaded", id=msg_id)
            return 0
        else:
            emit("error", id=msg_id, stage="protocol", kind="bad_request", message=f"unknown op {op!r}", hint="")
    engine.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
