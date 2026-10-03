"""Model-free tests for sa3_mlx_worker.py. Plain python3 is enough (no mlx, no numpy, no weights):
    python3 worker/test_sa3_mlx_worker.py        (or: python3 -m unittest worker/test_sa3_mlx_worker.py)

The worker runs as a subprocess against a fake "stable-audio-3 dir" (empty weight files, symlinked the way the Hugging Face cache
does it). Generation tests swap the MLX `Engine` for a stand-in with the same interface, so the protocol, validation, file
writing and error mapping are the real code.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "sa3_mlx_worker.py"
REVISION = "da6edc54ddba10bfd79a077102ded687f80e882b"
WEIGHTS = ("dit_sm-sfx_f16.npz", "t5gemma_f16.npz", "same_s_decoder_f32.npz")

FAKE_ENGINE = r'''
import array, math, os, sys

class FakeItem:
    def __init__(self, frames, seed, nan):
        self.frames, self.seed, self.nan = frames, seed, nan
    def all_finite(self):
        return not self.nan
    def to_f32le_interleaved(self):
        a = array.array("f")
        for i in range(self.frames):
            v = float("nan") if (self.nan and i == 3) else 0.1 * math.sin(i * 0.01 * (1 + self.seed % 7))
            a.extend([v, -v])
        return a.tobytes()

class FakeEngine:
    def __init__(self, sa3_dir):
        self.loaded = False
        self.mode = os.environ.get("FAKE_ENGINE_MODE", "ok")
        self.calls = []
    def load(self):
        if self.mode == "load_missing":
            raise FileNotFoundError("[Errno 2] No such file or directory: 't5gemma_f16.npz'")
        if self.mode == "load_oom":
            raise RuntimeError("[metal::malloc] Attempting to allocate 9999 bytes which is greater than the maximum allowed buffer size")
        self.loaded = True
    def unload(self):
        self.loaded = False
    def reset_peak(self): pass
    def clear_cache(self): pass
    def peak_mb(self): return 2400.5
    def active_mb(self): return 1600.0
    def generate(self, prompt, seconds, steps, seeds, progress, timings):
        self.calls.append((prompt, seconds, steps, list(seeds)))
        print("chatter from the model code")  # must land on stderr
        os.write(1, b"C-level chatter\n")
        progress("condition", 0, 1, None)
        progress("sample", 0, steps, None)
        if self.mode == "exc" and prompt.startswith("boom"):
            raise ValueError("kaboom")
        if self.mode == "oom" and prompt.startswith("boom"):
            raise RuntimeError("[metal::malloc] Attempting to allocate 9999 bytes which is greater than the maximum allowed buffer size")
        for i in range(1, steps + 1):
            progress("sample", i, steps, None)
        timings.update(condition=0.01, sample=0.02, decode=0.03)
        frames = int(round(seconds * 44100))
        out = []
        for i, s in enumerate(seeds):
            progress("decode", i, len(seeds), i)
            out.append(FakeItem(frames, s, self.mode == "nan" and prompt.startswith("boom")))
        return out
'''


def make_sa3_dir(root, missing=(), dangling=()):
    """<root>/models/mlx/<weights>: symlinks into a fake HF snapshot, as in the real checkout."""
    root = Path(root)
    snap = root / "hf" / "snapshots" / REVISION / "MLX"
    blobs = root / "hf" / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir(parents=True)
    mlx_dir = root / "sa3" / "models" / "mlx"
    mlx_dir.mkdir(parents=True)
    for i, name in enumerate(WEIGHTS):
        if name in missing:
            continue
        blob = blobs / f"blob{i}"
        blob.write_bytes(b"x")
        (snap / name).symlink_to(blob)
        os.symlink(snap / name, mlx_dir / name)
        if name in dangling:
            blob.unlink()
    return root / "sa3"


def run_worker(sa3_dir, lines, fake=True, mode="ok", extra_code="", timeout=60):
    """Returns (events, stderr, returncode). Raises if stdout holds anything but JSON object lines."""
    code = "import sys, sa3_mlx_worker as w\n"
    if fake:
        code += FAKE_ENGINE + "\nw.Engine = FakeEngine\n"
    code += extra_code + "\nsys.exit(w.main(['sa3_mlx_worker.py', sys.argv[1]]))\n"
    proc = subprocess.run(
        [sys.executable, "-u", "-c", code, str(sa3_dir)], input="\n".join(lines) + ("\n" if lines else ""),
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(HERE), "FAKE_ENGINE_MODE": mode},
    )
    events = []
    for line in proc.stdout.splitlines():
        events.append(json.loads(line))  # a non-JSON line fails the test here
        assert isinstance(events[-1], dict) and "event" in events[-1], line
    return events, proc.stderr, proc.returncode


def req(op, **fields):
    return json.dumps({"op": op, **fields})


def gen(job_id="j1", **over):
    fields = {"id": job_id, "prompt": "a short meep", "seconds": 1.0, "steps": 8, "seeds": [11, 12]}
    fields.update(over)
    return req("generate", **fields)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sa3 = make_sa3_dir(self.root)
        self.out = self.root / "out"
        self.out.mkdir()

    def tearDown(self):
        for p in self.root.rglob("*"):
            try:
                os.chmod(p, 0o755 if p.is_dir() else 0o644)
            except OSError:
                pass
        self._tmp.cleanup()

    def g(self, job_id="j1", **over):
        over.setdefault("out_dir", str(self.out))
        return gen(job_id, **over)


class Handshake(Base):
    def test_hello_ping_unknown_op_and_unload(self):
        events, _, code = run_worker(self.sa3, [req("ping", id="a"), "nope", req("x", id="b"), "[1]", req("unload", id="u")])
        self.assertEqual(code, 0)
        self.assertEqual([e["event"] for e in events], ["hello", "pong", "error", "error", "error", "unloaded"])
        hello = events[0]
        self.assertEqual(hello["protocol"], 1)
        self.assertIs(hello["weights_ok"], True)
        self.assertEqual(hello["revision"], REVISION)
        self.assertIsInstance(hello["pid"], int)
        self.assertTrue(hello["python"])
        self.assertIn("mlx", hello)
        self.assertEqual(events[1]["id"], "a")
        self.assertEqual([(e["stage"], e["kind"]) for e in events[2:5]], [("protocol", "bad_request")] * 3)
        self.assertEqual(events[3]["id"], "b")
        self.assertEqual(events[5]["id"], "u")

    def test_eof_ends_the_worker(self):
        events, _, code = run_worker(self.sa3, [])
        self.assertEqual((code, [e["event"] for e in events]), (0, ["hello"]))

    def test_missing_weight_is_one_error_and_exit_3(self):
        for kwargs in ({"missing": (WEIGHTS[1],)}, {"dangling": (WEIGHTS[0], WEIGHTS[2])}):
            with tempfile.TemporaryDirectory() as d:
                sa3 = make_sa3_dir(d, **kwargs)
                events, _, code = run_worker(sa3, [req("ping", id="a")])
                self.assertEqual(code, 3, kwargs)
                self.assertEqual(len(events), 1, events)
                e = events[0]
                self.assertEqual((e["event"], e["kind"], e["stage"]), ("error", "missing_weights", "startup"))
                for name in set(kwargs.get("missing", ())) | set(kwargs.get("dangling", ())):
                    self.assertIn(name, e["message"])

    def test_usage_error_without_argument(self):
        proc = subprocess.run([sys.executable, str(WORKER)], capture_output=True, text=True, timeout=30)
        self.assertEqual((proc.returncode, proc.stdout), (2, ""))

    def test_command_line_names_the_worker(self):
        self.assertIn("sa3_mlx_worker", str(WORKER))

    def test_library_prints_never_reach_the_protocol_stream(self):
        code = (
            "import os\n"
            "print('chatter from a library')\n"
            "os.write(1, b'C-level chatter\\n')\n"
            "w.emit('probe', ok=True)\n"
        )
        events, stderr, _ = run_worker(self.sa3, [], fake=False, extra_code=code)
        self.assertEqual([e["event"] for e in events], ["probe", "hello"])
        self.assertIn("chatter from a library", stderr)
        self.assertIn("C-level chatter", stderr)


class Validation(Base):
    def rejected(self, **over):
        """The request is a bad_request, and the worker stays alive (a ping after it is answered)."""
        events, _, code = run_worker(self.sa3, [self.g("v1", **over), req("ping", id="after")])
        self.assertEqual(code, 0)
        names = [e["event"] for e in events]
        self.assertEqual(names, ["hello", "error", "pong"], (over, events))
        err = events[1]
        expected_id = over["id"] if "id" in over and isinstance(over["id"], str) else ("" if "id" in over else "v1")
        self.assertEqual((err["kind"], err["stage"], err["id"]), ("bad_request", "request", expected_id), over)
        self.assertTrue(err["message"])
        return err

    def test_bad_requests(self):
        for over in (
            {"prompt": ""}, {"prompt": "   "}, {"prompt": None}, {"prompt": 5},
            {"seconds": 0.09}, {"seconds": 120.01}, {"seconds": 0}, {"seconds": -1}, {"seconds": "1"}, {"seconds": None},
            {"seconds": True}, {"seconds": float("nan")},
            {"steps": 0}, {"steps": 51}, {"steps": 1.5}, {"steps": "8"}, {"steps": True},
            {"seeds": []}, {"seeds": list(range(9))}, {"seeds": [-1]}, {"seeds": [2**63]}, {"seeds": [True]},
            {"seeds": [1.5]}, {"seeds": ["1"]}, {"seeds": [3, 3]}, {"seeds": 7}, {"seeds": None},
            {"out_dir": str(self.root / "nope")}, {"out_dir": ""}, {"out_dir": 5}, {"out_dir": __file__},
            {"id": ""}, {"id": "../x"}, {"id": "a/b"}, {"id": 7}, {"id": "x" * 65},
        ):
            with self.subTest(over=over):
                self.rejected(**over)

    def test_missing_fields_are_bad_requests(self):
        for field in ("prompt", "seconds", "seeds", "out_dir"):
            with self.subTest(field=field):
                fields = {"id": "m", "prompt": "p", "seconds": 1.0, "seeds": [1], "out_dir": str(self.out)}
                del fields[field]
                events, _, _ = run_worker(self.sa3, [req("generate", **fields)])
                self.assertEqual(events[1]["kind"], "bad_request")

    def test_missing_id_reports_an_empty_id(self):
        events, _, _ = run_worker(self.sa3, [req("generate", prompt="p", seconds=1, seeds=[1], out_dir=str(self.out))])
        self.assertEqual((events[1]["kind"], events[1]["id"]), ("bad_request", ""))

    def test_boundary_values_are_accepted(self):
        lines = [
            self.g("b1", seconds=0.1, steps=1, seeds=[0]),
            self.g("b2", seconds=120, steps=50, seeds=[2**63 - 1] + list(range(7))),
            self.g("b3", seconds=7),  # an int is a number
        ]
        events, _, code = run_worker(self.sa3, lines)
        self.assertEqual(code, 0)
        results = [e for e in events if e["event"] == "result"]
        self.assertEqual([e["id"] for e in results], ["b1", "b2", "b3"], events)
        self.assertEqual([e["frames"] for e in results], [4410, 5292000, 308700])
        self.assertEqual(len(results[1]["files"]), 8)

    def test_a_rejected_request_does_not_trigger_a_load(self):
        events, _, _ = run_worker(self.sa3, [self.g("v1", prompt=""), self.g("v2")])
        self.assertEqual([e["event"] for e in events[:2]], ["hello", "error"])
        self.assertEqual(sum(e["event"] == "loading" for e in events), 1)
        self.assertEqual(events[2]["event"], "loading")
        self.assertEqual(events[2]["id"], "v2")


class Generate(Base):
    def test_event_sequence_result_and_files(self):
        events, stderr, code = run_worker(self.sa3, [self.g("j1", seeds=[5, 6, 7], seconds=0.5), self.g("j2", seeds=[9])])
        self.assertEqual(code, 0)
        by_job = {"j1": [], "j2": []}
        for e in events[1:]:
            by_job[e["id"]].append(e)
        j1, j2 = by_job["j1"], by_job["j2"]
        self.assertEqual(j1[0]["event"], "loading")  # cold
        self.assertNotIn("loading", [e["event"] for e in j2])  # warm
        self.assertEqual([e["event"] for e in j1[1:-1]], ["progress"] * (len(j1) - 2))
        stages = [e["stage"] for e in j1 if e["event"] == "progress"]
        self.assertEqual(stages[0], "condition")
        self.assertEqual(sorted(set(stages)), ["condition", "decode", "sample"])
        self.assertEqual(stages, sorted(stages, key=["condition", "sample", "decode"].index))
        sample = [(e["step"], e["total"]) for e in j1 if e.get("stage") == "sample"]
        self.assertEqual(sample, [(i, 8) for i in range(9)])
        decode = [(e["step"], e["total"], e["item"]) for e in j1 if e.get("stage") == "decode"]
        self.assertEqual(decode, [(0, 3, 0), (1, 3, 1), (2, 3, 2)])
        for e in j1:
            if e["event"] == "progress":
                self.assertEqual(set(e), {"event", "id", "stage", "step", "total", "item"})
        result = j1[-1]
        self.assertEqual(result["event"], "result")
        self.assertEqual((result["sample_rate"], result["channels"], result["frames"]), (44100, 2, 22050))
        self.assertEqual(result["seeds"], [5, 6, 7])
        self.assertEqual(result["files"], [str(self.out / f"j1_{s}.f32") for s in (5, 6, 7)])
        self.assertEqual(set(result["timings"]), {"load", "condition", "sample", "decode", "total"})
        self.assertEqual((result["peak_mlx_mb"], result["active_mlx_mb"]), (2400.5, 1600.0))
        self.assertIn("peak_rss_mb", result)
        self.assertGreaterEqual(result["timings"]["total"], result["timings"]["load"])
        self.assertEqual(j2[-1]["timings"]["load"], 0.0)
        for path in result["files"]:
            self.assertTrue(os.path.isabs(path))
            self.assertEqual(os.path.getsize(path), 22050 * 2 * 4)
        self.assertEqual(sorted(os.listdir(self.out)), ["j1_5.f32", "j1_6.f32", "j1_7.f32", "j2_9.f32"])  # no .partial left
        # raw little-endian float32, interleaved L/R (the stand-in writes R = -L)
        raw = Path(result["files"][0]).read_bytes()
        left, right = struct.unpack("<ff", raw[8:16])
        self.assertEqual(right, -left)
        self.assertNotEqual(left, 0.0)
        self.assertIn("chatter from the model code", stderr)

    def test_default_steps_and_relative_out_dir_become_absolute(self):
        line = json.dumps({"op": "generate", "id": "d", "prompt": "p", "seconds": 1, "seeds": [1], "out_dir": "."})
        proc = subprocess.run(
            [sys.executable, "-u", "-c",
             "import sys, sa3_mlx_worker as w\n" + FAKE_ENGINE + "\nw.Engine = FakeEngine\nsys.exit(w.main(['x', sys.argv[1]]))",
             str(self.sa3)],
            input=line + "\n", capture_output=True, text=True, cwd=self.out, env={**os.environ, "PYTHONPATH": str(HERE)},
        )
        events = [json.loads(l) for l in proc.stdout.splitlines()]
        result = events[-1]
        self.assertEqual(result["event"], "result", events)
        self.assertEqual(result["files"], [os.path.join(os.path.realpath(self.out), "d_1.f32")])
        sample_total = [e["total"] for e in events if e.get("stage") == "sample"][0]
        self.assertEqual(sample_total, 8)

    def test_seeds_reach_the_engine_unchanged_and_in_order(self):
        events, _, _ = run_worker(self.sa3, [self.g("s1", seeds=[2**63 - 1, 0, 42], prompt="  padded  ")])
        result = events[-1]
        self.assertEqual(result["seeds"], [2**63 - 1, 0, 42])
        self.assertEqual([Path(p).name for p in result["files"]], [f"s1_{s}.f32" for s in (2**63 - 1, 0, 42)])

    def test_nan_output_writes_nothing_and_the_worker_survives(self):
        lines = [self.g("n1", prompt="boom nan"), req("ping", id="alive"), self.g("n2")]
        events, stderr, code = run_worker(self.sa3, lines, mode="nan")
        self.assertEqual(code, 0)
        err = next(e for e in events if e["event"] == "error")
        self.assertEqual((err["id"], err["kind"], err["stage"]), ("n1", "nan_output", "decode"))
        self.assertIn("seed", err["message"])
        self.assertTrue(err["hint"])
        self.assertIn("pong", [e["event"] for e in events])
        self.assertEqual(events[-1]["event"], "result")
        self.assertEqual(sorted(os.listdir(self.out)), ["n2_11.f32", "n2_12.f32"])  # nothing from n1, not even .partial

    def test_out_of_memory_is_mapped(self):
        events, _, _ = run_worker(self.sa3, [self.g("o1", prompt="boom")], mode="oom")
        err = events[-1]
        self.assertEqual((err["event"], err["kind"], err["id"]), ("error", "out_of_memory", "o1"))
        self.assertEqual(err["stage"], "sample")
        self.assertIn("close", err["hint"])
        events, _, _ = run_worker(self.sa3, [self.g("o2")], mode="load_oom")
        self.assertEqual((events[-1]["kind"], events[-1]["stage"]), ("out_of_memory", "loading"))

    def test_worker_exception_keeps_the_worker_alive(self):
        lines = [self.g("e1", prompt="boom"), req("ping", id="p"), self.g("e2")]
        events, stderr, code = run_worker(self.sa3, lines, mode="exc")
        self.assertEqual(code, 0)
        err = next(e for e in events if e["event"] == "error")
        self.assertEqual((err["kind"], err["stage"], err["id"]), ("worker_exception", "sample", "e1"))
        self.assertEqual(err["message"], "ValueError: kaboom")
        self.assertIn("Traceback", stderr)  # traceback goes to stderr, never the protocol stream
        self.assertEqual([e["event"] for e in events if e["event"] in ("pong", "result")], ["pong", "result"])
        self.assertEqual(events[-1]["id"], "e2")

    def test_weight_vanishing_after_startup_is_missing_weights(self):
        events, _, _ = run_worker(self.sa3, [self.g("w1")], mode="load_missing")
        err = events[-1]
        self.assertEqual((err["kind"], err["stage"]), ("missing_weights", "loading"))

    def test_unwritable_out_dir_is_a_worker_exception_without_partials(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores directory permissions")
        os.chmod(self.out, 0o500)
        events, _, _ = run_worker(self.sa3, [self.g("x1")])
        err = events[-1]
        self.assertEqual((err["event"], err["kind"], err["stage"]), ("error", "worker_exception", "write"))
        self.assertEqual(os.listdir(self.out), [])

    def test_unload_after_a_job_frees_and_exits_zero(self):
        events, _, code = run_worker(self.sa3, [self.g("u1"), req("unload", id="bye"), req("ping", id="never")])
        self.assertEqual(code, 0)
        self.assertEqual(events[-1], {"event": "unloaded", "id": "bye"})
        self.assertNotIn("never", json.dumps(events))

    def test_a_generate_after_unload_would_not_be_read(self):
        # unload ends the worker: the server starts a new process for the next job
        events, _, _ = run_worker(self.sa3, [req("unload", id="a"), self.g("late")])
        self.assertEqual([e["event"] for e in events], ["hello", "unloaded"])


class Units(Base):
    def run_snippet(self, code):
        events, stderr, rc = run_worker(self.sa3, [], fake=False, extra_code=code)
        self.assertEqual(rc, 0, stderr)
        return events

    def test_revision_comes_from_the_snapshot_symlink(self):
        events = self.run_snippet(f"w.emit('r', value=w.check_weights({str(self.sa3)!r}))")
        value = next(e for e in events if e["event"] == "r")["value"]
        self.assertEqual(value, [[], REVISION])

    def test_a_plain_file_has_no_revision(self):
        with tempfile.TemporaryDirectory() as d:
            mlx_dir = Path(d, "models", "mlx")
            mlx_dir.mkdir(parents=True)
            for name in WEIGHTS:
                (mlx_dir / name).write_bytes(b"x")
            events = self.run_snippet(f"w.emit('r', value=w.check_weights({d!r}))")
            self.assertEqual(next(e for e in events if e["event"] == "r")["value"], [[], None])

    def test_error_text_is_classified(self):
        events = self.run_snippet(
            "for e in (RuntimeError('[metal::malloc] Attempting to allocate 5 bytes'), MemoryError(), ValueError('x'),\n"
            "          FileNotFoundError('gone')):\n"
            "    m = w.classify_exception('loading', e)\n"
            "    w.emit('c', kind=m.kind if m else None)\n"
        )
        self.assertEqual([e["kind"] for e in events if e["event"] == "c"],
                         ["out_of_memory", "out_of_memory", None, "missing_weights"])


if __name__ == "__main__":
    unittest.main()
