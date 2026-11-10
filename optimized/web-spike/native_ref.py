"""Native (Python ai_edge_litert) reference run for the browser spike.

Runs the same text-to-audio pipeline as optimized/tflite/scripts/sa3_tflite.py (small DiT w8a8 +
SAME-S decoder w8a8 + T5Gemma fp16) and dumps every input and intermediate the browser needs for
a component-by-component parity check:

    ids, mask          tokenized prompt (the browser spike skips SentencePiece)
    t5_hidden          T5Gemma output
    x0, step_noise     initial and per-step noise (so the browser doesn't port numpy's RNG)
    sigmas             pingpong schedule
    v_step1            DiT velocity for (x0, sigmas[0]) given t5_hidden -> isolates one DiT forward
    gc_step1           adaLN global conditioning from the 'gcond' signature at sigmas[0]
    latents            final latents after sampling
    audio              decoded audio (2, samples), trimmed to --seconds

Arrays are written as raw little-endian files plus manifest.json (name -> dtype, shape, file).
Usage (from optimized/tflite, with its venv):
    python ../web-spike/native_ref.py --out ../web-spike/parity --seconds 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

TFLITE_DIR = Path(__file__).resolve().parent.parent / "tflite"
sys.path.insert(0, str(TFLITE_DIR))
sys.path.insert(0, str(TFLITE_DIR / "scripts"))

from models.defs import tflite_pipeline as P  # noqa: E402
from rung_decoder import RungDecoder  # noqa: E402
from rung_dit import RungDiT  # noqa: E402
from weights import dec_rel, dit_rel, ensure_local  # noqa: E402

T5_REL = "models/tflite/t5gemma/encoder_fp16.tflite"
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
SAME_S_TRIM = 16


def valid_T_lat(seconds):
    return max(1, int(np.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT)))


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="Warm lo-fi hip hop beat with dusty drums, mellow Rhodes chords and vinyl crackle, 85 BPM")
    ap.add_argument("--dit", default="sm-music")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    timings = {}

    def dump(name, arr):
        arr = np.ascontiguousarray(arr)
        fname = f"{name}.bin"
        arr.astype(arr.dtype.newbyteorder("<")).tofile(out / fname)
        manifest[name] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "file": fname}

    os.chdir(TFLITE_DIR)
    T_lat = valid_T_lat(args.seconds)

    t0 = time.perf_counter()
    ids, mask = P.Tokenizer()(args.prompt)
    t5 = P.T5GemmaTFLite(ensure_local(T5_REL), args.threads)
    timings["t5_load_ms"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    t5_hidden = t5(ids, mask)
    timings["t5_run_ms"] = (time.perf_counter() - t0) * 1000
    del t5

    x0, step_noise = P.make_noise(T_lat, args.steps, args.seed)
    sigmas = P.build_pingpong_schedule(args.steps, 1.0)

    t0 = time.perf_counter()
    dit = RungDiT(ensure_local(dit_rel(args.dit, "w8a8")), T_lat, t5_hidden, mask.astype(np.float32),
                  args.seconds, threads=args.threads)
    timings["dit_load_ms"] = (time.perf_counter() - t0) * 1000
    v_step1 = dit(x0, float(sigmas[0]))
    gc_step1 = dit._compute_gc(float(sigmas[0]))
    dit.n_fwd = 0
    step_ms = []
    last = [time.perf_counter()]

    def on_step(i, total):
        now = time.perf_counter()
        step_ms.append((now - last[0]) * 1000)
        last[0] = now

    t0 = time.perf_counter()
    last[0] = t0
    latents = P.sample(dit, x0, step_noise, sigmas, None, None, on_step=on_step)
    timings["dit_sample_ms"] = (time.perf_counter() - t0) * 1000
    timings["dit_step_ms"] = step_ms
    timings["dit_rung"] = dit.R
    del dit

    t0 = time.perf_counter()
    dec = RungDecoder(ensure_local(dec_rel("same-s", "w8a8")), threads=args.threads, trim=SAME_S_TRIM)
    timings["dec_load_ms"] = (time.perf_counter() - t0) * 1000
    dec_lat = latents
    if T_lat % 2:
        dec_lat = np.concatenate([latents, latents[:, :, -1:]], axis=2)
    t0 = time.perf_counter()
    audio = dec.decode(dec_lat)[:, :, : T_lat * SAMPLES_PER_LATENT]
    timings["dec_run_ms"] = (time.perf_counter() - t0) * 1000
    timings["dec_rung"] = dec._best_rung(dec_lat.shape[2])
    audio = audio[0][:, : int(round(args.seconds * SAMPLE_RATE))]

    for name, arr in [("ids", ids), ("mask", mask), ("t5_hidden", t5_hidden), ("x0", x0),
                      ("step_noise", np.stack(step_noise)), ("sigmas", sigmas), ("v_step1", v_step1), ("gc_step1", gc_step1),
                      ("latents", latents), ("audio", audio.astype(np.float32))]:
        dump(name, arr)
    P.save_wav(out / "native.wav", audio)

    timings["peak_rss_mb"] = peak_rss_mb()
    meta = {"prompt": args.prompt, "dit": args.dit, "seconds": args.seconds, "seed": args.seed,
            "steps": args.steps, "threads": args.threads, "T_lat": T_lat,
            "trim": SAME_S_TRIM, "timings": timings, "arrays": manifest}
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
