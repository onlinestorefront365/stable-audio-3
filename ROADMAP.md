# Roadmap

Work planned for this fork of [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3).
The fork started identical to upstream `main` (Nov 2026). Items are ordered so the early
phases pay off whatever the end goal is: personal use, contributing upstream, or a product.

**Status:** ✅ done · 🚧 in progress · ⬜ not started

**Ground rules**

- Keep changes small and upstream-mergeable: one concern per commit, no drive-by reformatting
  of the directories `pyproject.toml` excludes from ruff.
- Every behavior change ships with a test that runs on CPU without model downloads
  (mock the model, as `tests/test_cli.py` does).
- Security fixes that also affect upstream are reported to Stability AI privately before
  anything about them is posted publicly.
- Sync from upstream `main` regularly; fixes that make sense upstream go back as PRs.

---

## Phase 0: Security and correctness

These come first. P0.1 and P0.2 together let anyone holding the Gradio share link run
commands on the host.

| ID | Problem | Where | Status |
|---|---|---|---|
| P0.1 | **Shell injection in Gradio export.** MP3/FLAC/M4A export ran `ffmpeg` through `shell=True` with a filename built from the prompt; `condense_prompt` left `$(...)` and backticks intact. | `stable_audio_3/interface/diffusion_cond.py` | ✅ |
| P0.2 | **Gradio is always public and never authenticated.** `launch(share=True)` is hard-coded; `--username`/`--password` are parsed but never passed to `launch()`. | `run_gradio.py` | 🚧 |
| P0.3 | **The random seed is lost.** With `seed=-1` the seed picked inside `generate()` is never returned, printed, or shown; the Gradio filename reads `.-1`. | `stable_audio_3/model.py`, `cli.py`, `interface/diffusion_cond.py` | 🚧 |
| P0.4 | **Gradio output files collide.** Files are written to the working directory under fixed names (e.g. `output.wav`), so concurrent users overwrite each other. | `interface/diffusion_cond.py` | ✅ |
| P0.5 | **`run_gradio.py` flags that do nothing.** `--model-half` can never be turned off; `--model-config`, `--ckpt-path`, `--pretransform-ckpt-path` are silently ignored. Fix: make `--no-model-half` work and fail loudly on the unsupported flags (loading local checkpoints is P1.9). | `run_gradio.py` | 🚧 |
| P0.6 | **`from_pretrained` accepts autoencoder names.** `all_models` includes `same-s`/`same-l`, so `StableAudioModel.from_pretrained("same-s")` passes validation and fails later with a confusing error. | `stable_audio_3/model.py` | 🚧 |
| P0.7 | **Gradio batches are glued into one file.** `rearrange("b d n -> d (b n)")` concatenates variations end to end instead of returning one file per variation. | `interface/diffusion_cond.py` | ⬜ |
| P0.8 | **41 MB of generated spectrograms are committed.** 222 PNGs under `optimized/tensorRT/output/` are tracked despite `.gitignore`. Untrack them (`git rm --cached`). | `optimized/tensorRT/output/` | ⬜ |
| P0.9 | **CI never runs the core test suite.** Only lint, one MLX test and a TFLite smoke check run. Add a CPU job that runs the mocked unit tests (`test_cli.py`, `test_gradio_interface.py`, `test_run_gradio.py`, …). | `.github/workflows/` | ⬜ |
| P0.10 | **Report P0.1/P0.2 upstream privately.** | — | ⬜ |

---

## Phase 1: Quick wins on existing building blocks

Each of these reuses something that already works (inpainting, continuation, the low-level
`conditioning_tensors` path, LoRA strength controls), so none needs new training.

| ID | Feature | Approach | Done when |
|---|---|---|---|
| P1.1 | **Reproducibility metadata** | Write prompt, negative prompt, seed, model, steps, CFG and LoRA settings into the output file (WAV `LIST/INFO` or a sidecar JSON; ID3 for MP3). Add "load settings from file" to the UI and `--from-file` to the CLI. | Re-running from a saved file reproduces the audio bit-for-bit on the same device. |
| P1.2 | **CLI parity with the UI** | `--lora path:0.8` per-adapter strength; `--sampler`, `--apg-scale`, `--dist-shift`, `--cfg-interval`; `--format mp3/flac`; `--prompts-file` (CSV/JSONL) for batch runs. | Every Gradio control has a CLI flag; covered by mocked tests. |
| P1.3 | **Post-processing** | Optional loudness normalization to a LUFS target, fade in/out, trailing-silence trim. Shared helper used by CLI, UI and API. | Unit tests on synthetic signals. |
| P1.4 | **Seamless loop mode** | Choose bars + BPM → duration. Generate, rotate the audio by half its length, inpaint the seam region, rotate back. | Loop plays end-to-start with no audible click; seam RMS discontinuity below a threshold in a test. |
| P1.5 | **Coherent endless stream** | Replace Infinite Radio's independent tracks with chained continuation: inpaint with the last N seconds as context, crossfade the joins. | 10+ minutes of continuous audio without hard cuts. |
| P1.6 | **Prompt morphing** | Encode two prompts, interpolate the conditioning tensors, generate through the existing `conditioning_tensors` argument of `generate()`. Optional time-varying morph via chained continuation. | `generate(prompt_a, prompt_b, mix=0.0..1.0)` API plus UI slider. |
| P1.7 | **Waveform region picker** | Select inpaint regions on the waveform instead of typing comma-separated seconds. | Regions selected in the UI match the seconds passed to `generate()`. |
| P1.8 | **Seed reuse in the UI** | "Reuse last seed" button and a visible "seed used" field (seed is already surfaced in the filename and spectrogram caption by P0.3). | One click re-runs with the previous seed. |
| P1.9 | **Load local checkpoints** | `StableAudioModel.from_local(config, ckpt)` plus a separate autoencoder checkpoint; wire `run_gradio.py --model-config/--ckpt-path/--pretransform-ckpt-path` and matching CLI flags. | A fine-tuned full checkpoint loads without uploading it to the Hub. |

---

## Phase 2: Productization

What turns the repo into something other people or apps use. Order depends on the end goal.

| ID | Feature | Notes |
|---|---|---|
| P2.1 | **HTTP API server** | FastAPI service: job queue, API-key auth, GPU batching, object storage for outputs, health/metrics endpoints. Today the only server is Gradio. |
| P2.2 | **Multi-tab Gradio** | Autoencoder tab (encode/decode, latent PCA view; `aeiou.pca_point_cloud` exists but no UI uses it), LoRA training tab, generation history/gallery. |
| P2.3 | **Deployment recipe** | Dockerfile(s) for CPU (small models) and CUDA (medium), with model-cache volume and a one-command run. |
| P2.4 | **DAW integration beyond Ableton on Mac** | The Ableton bridge only works with the MLX runtime. Options: port it to the PyTorch/TensorRT runtimes, or a VST3/CLAP plugin that talks to the P2.1 API. |
| P2.5 | **Usage limits and abuse controls** | Per-key rate limits, max duration/batch per request, prompt logging policy. Needed before any public deployment. |

---

## Phase 3: Research (needs training)

| ID | Feature | Notes |
|---|---|---|
| P3.1 | **Numeric BPM / key conditioning** | The architecture already supports numeric conditioners (`NumberConditioner`, used for duration). Needs a fine-tune with BPM/key labels. |
| P3.2 | **Stem-for-mix generation** | Generate a part (e.g. a bassline) that fits an existing mix. Needs a conditioning scheme and training data with aligned stems. |

---

## Open decisions

These change the order of Phase 2 and beyond; Phases 0–1 are useful regardless.

- **End goal:** personal tool, upstream contributions, or a product. A product makes P2.1, P2.3
  and P2.5 the next priorities.
- **Licensing:** the models are under the [Stability AI Community License](https://stability.ai/license),
  which limits commercial use by revenue. Review it before shipping anything paid.
- **Hosting target:** CPU-only (small models, TFLite runtime) vs. NVIDIA GPU (medium, TensorRT)
  decides which runtime the API server wraps.
