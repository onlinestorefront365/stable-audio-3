# In-browser inference spike (ROADMAP P2.6)

Can the small Stable Audio 3 models (`small-music` / `small-sfx`) generate audio entirely in a web
browser, on the visitor's CPU? This spike ran the existing `optimized/tflite` int8 models with
[LiteRT.js](https://www.npmjs.com/package/@litertjs/core) 2.6.0 (WebAssembly + XNNPACK) in headless
Chromium. Every stage was checked against the native Python TFLite runtime on identical inputs.

**Result: yes, with three model-file changes.** A 30 s clip generated fully in the browser in
**15.1 s** on 4 threads. The output matches the native render: the same piece, with the small
differences you'd expect from int8 maths. Peak memory is ~2 GB, so this is desktop-only.

## Results

Measured on a cloud VM: 4 vCPU Intel Xeon @ 2.1 GHz (AVX-512 VNNI + AMX), 15 GB RAM, headless
Chromium 153, no GPU. Same prompt, seed 42, 8 steps, 30 s, `sm-music` w8a8 DiT, SAME-S w8a8
decoder.

| Stage (30 s clip) | Browser, 4 threads | Browser, 1 thread | Native Python, 4 threads |
|---|---|---|---|
| T5Gemma text encoder | 0.49 s | — | 0.97 s |
| DiT, 8 steps | 11.5 s (1.44 s/step) | 40.8 s (5.1 s/step) | 2.08 s (0.26 s/step) |
| SAME-S decode | 3.1 s | 12.0 s | 0.79 s |
| **Generation total** | **15.1 s** | ≈ 53 s | 3.8 s |
| Model compile (after download) | 1.8 s | 1.6 s | — |
| Peak memory | 1.99 GB (renderer process) | 1.67 GB | 2.07 GB |

- The browser is 4–6× slower than native on this VM. That gap is wider than on a typical laptop,
  because native XNNPACK uses this Xeon's AMX/VNNI int8 instructions and WebAssembly can't.
- Multithreading needs the page served with `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`. Without them it is single-threaded and ~3.5× slower.
- The model-load times above use a local server. Over a real connection the first visit is
  dominated by downloading ~710 MB (see below).
- WebGPU was not tested (this VM has no GPU), and neither were Safari, Firefox or mobile.

### Parity with native

| Check | cos | SNR |
|---|---|---|
| Pingpong schedule (JS port) | 1.0 | exact |
| T5 output (split model, see below) | 1.0 | 117 dB |
| `gcond` (adaLN conditioning) | 1.0 | 140 dB |
| One DiT forward, browser int8 vs native int8 | 0.9946 | 19.7 dB |
| One DiT forward, browser int8 vs **fp32** | 0.9905 | 17.2 dB |
| One DiT forward, native int8 vs **fp32** | 0.9921 | 18.0 dB |
| Decoder on identical latents | 0.9997 | 32.3 dB |
| Final audio, end to end | 0.963 | 11.3 dB |

The browser's int8 kernels round differently from native XNNPACK's, so it lands 0.8 dB further
from fp32 than native int8 does: the same quality tier. The 8-step sampler amplifies small
differences, so the end-to-end audio is a slightly different render of the same piece. Peak and
loudness match to 0.003 dB, spectral centroid is 244 vs 245 Hz, and mean per-band energy differs by
0.31 dB. `out/spectrograms.png` shows the two side by side.

## What had to change

1. **LiteRT.js caps the WASM heap at 2 GB.** All four builds (`litert_wasm{,_threaded,_jspi,_compat}_internal.wasm`)
   declare a 32,768-page maximum, so the 4 GB wasm32 limit isn't reachable.
2. **Multi-rung files overflow that heap.** LiteRT.js prepares every signature at compile time
   and, with no XNNPACK weight cache in the browser, packs the shared weights once per signature.
   The 8-rung DiT fails with `failed to create XNNPACK runtime`. **Fix:** `slice_signatures.py`
   keeps only the signatures a render needs (`gcond` + one rung for the DiT, one rung for the
   decoder). The sliced files are bit-identical to the full ones natively.
3. **T5Gemma doesn't fit either.** Its 256,000 × 768 fp16 embedding table (197M of 282M weights)
   is expanded to fp32 (786 MB) by an in-graph `DEQUANTIZE`. **Fix:** `t5_split.py` removes the
   lookup. The page fetches only the rows a prompt uses (HTTP Range requests, 1.5 KB per token,
   about 35 KB for a typical prompt) and runs a 163 MB encoder body. Output is bit-identical
   natively, and the browser runs T5 2× faster than native because it no longer expands the table.

### Download size (first visit, then cacheable)

| File | Size |
|---|---|
| DiT, one rung + `gcond` (`dit_w8a8_s416.tflite`) | 476 MB |
| T5Gemma body (`t5_body_fp16.tflite`) | 163 MB |
| SAME-S decoder, one rung (`dec_w8a8_s256.tflite`) | 70 MB |
| LiteRT.js WASM | 11 MB |
| T5 embedding rows | ~35 KB per prompt |
| **Total** | **≈ 720 MB** (vs 1,142 MB for the original files) |

### Open problems before a product

- **Rung choice vs download.** The rungs share weights, so every single-rung DiT file carries all
  476 MB. Using the 2-minute rung (`s1292`) for every length fits in memory but is 3.6× slower for
  a 30 s clip (41.5 s vs 11.5 s). Options: slice the cached multi-rung file in the browser (a small
  flatbuffer rewrite in JS) before compiling, or ask upstream LiteRT.js for lazy per-signature
  compilation.
- **Memory.** About 2 GB peak rules out phones (iOS Safari kills tabs well below that). Passing
  `loadAndCompile` a stream instead of a whole `Uint8Array` might cut the JS-side copy.
- **Tokenizer.** The spike reused the tokenized prompt from Python. A product needs SentencePiece
  in JS (the T5Gemma `tokenizer.json` is on the Hub for tokenizers that read that format).
- **Real hardware.** The go/no-go bar was under 15 s for a 30 s clip on a mid-range laptop. This VM
  gave 15.1 s. A laptop with 8+ fast cores should be quicker, but that is an extrapolation until
  someone opens the page on one.
- **License.** Serving these weights to every visitor redistributes them; check the Stability AI
  Community License first.

## Reproduce

```bash
# 1. native reference + parity dumps (downloads ~1.1 GB from stabilityai/stable-audio-3-optimized)
cd optimized/tflite && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python ../web-spike/native_ref.py --out ../web-spike/parity --seconds 30 --seed 42 --threads 4

# 2. browser-sized model files
cd ../web-spike && npm ci
PY=../tflite/.venv/bin/python M=../tflite/models/tflite
$PY slice_signatures.py $M/sa3-sm-music/dit_w8a8.tflite sliced/dit_w8a8_s416.tflite gcond s416
$PY slice_signatures.py $M/same-s/dec_w8a8.tflite sliced/dec_w8a8_s256.tflite s256
$PY t5_split.py $M/t5gemma/encoder_fp16.tflite sliced

# 3. run (serve.mjs sends the cross-origin isolation headers; run_spike.mjs drives headless Chromium)
node serve.mjs &
NODE_PATH=$(npm root -g) node run_spike.mjs --threads 4 --sliced 1 --label t4_full
python compare_audio.py parity/audio.bin out/browser_audio_t4.f32 out/spectrograms.png  # needs torch + matplotlib
```

To try it on your own machine, run steps 1–2, start `node serve.mjs`, and open
`http://127.0.0.1:8765/?threads=8&sliced=1` (set threads to your core count). Results land in
`window.__spike.result`, with progress in the console.

| File | Purpose |
|---|---|
| `native_ref.py` | Native run; dumps inputs and intermediates for parity |
| `slice_signatures.py` | Keep only the named signatures of a rung `.tflite` |
| `t5_split.py` | Move T5Gemma's embedding lookup out of the graph |
| `spike.js`, `index.html` | The browser pipeline: T5 → gcond + DiT rung sampler → SAME-S rung decoder |
| `serve.mjs` | Static server with COOP/COEP and Range support, plus result upload |
| `run_spike.mjs` | Playwright driver; samples renderer RSS, writes `out/result_*.json` and WAVs |
| `compare_audio.py` | Spectrograms and stats, native vs browser |
