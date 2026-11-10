// Browser spike: run the optimized/tflite small-model pipeline with LiteRT.js and compare every
// stage against native_ref.py's dumps. Query params:
//   threads=N   use the multithreaded WASM build with N threads (needs cross-origin isolation)
//   accel=wasm  accelerator ('wasm' | 'webgpu')
//   sliced=1    load the single-rung files from slice_signatures.py
//   dit=URL     override the DiT file (e.g. /sliced/dit_w8a8_s1292.tflite)
//   stages=...  comma list of: t5,gcond,dit1,sample,decode_native,decode  (default: all)
import { loadAndCompile, loadLiteRt, Tensor } from "/litert/dist/index.js";

const q = new URLSearchParams(location.search);
const THREADS = Number(q.get("threads") || 0);
const ACCEL = q.get("accel") || "wasm";
const STAGES = new Set((q.get("stages") || "t5,gcond,dit1,sample,decode_native,decode").split(","));

const LATENT_CH = 256, COND_TOKENS = 256, COND_DIM = 768, MEM = 64, SPL = 4096, STITCH_TRIM = 8;
// sliced=1: single-rung files from slice_signatures.py (the full 8/9-rung files overflow the heap)
const SLICED = q.get("sliced") === "1";
const MODELS = {
  t5: "/models/t5gemma/encoder_fp16.tflite",
  dit: q.get("dit") || (SLICED ? "/sliced/dit_w8a8_s416.tflite" : "/models/sa3-sm-music/dit_w8a8.tflite"),
  dec: SLICED ? "/sliced/dec_w8a8_s256.tflite" : "/models/same-s/dec_w8a8.tflite",
};

const result = { env: {}, timings: {}, parity: {}, sizes: {}, notes: [], errors: [] };
const log = (...a) => { console.log("[spike]", ...a); };
const now = () => performance.now();
window.__spike = { result, done: false };

// ---------- helpers ----------
const CTORS = { float32: Float32Array, int32: Int32Array, int64: BigInt64Array, float64: Float64Array };

async function loadArray(manifest, name) {
  const m = manifest.arrays[name];
  const buf = await (await fetch(`/parity/${m.file}`)).arrayBuffer();
  return { data: new CTORS[m.dtype](buf), shape: m.shape };
}

function compare(a, b) {
  let dot = 0, na = 0, nb = 0, maxAbs = 0, err = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
    const d = Math.abs(x - y);
    if (d > maxAbs) maxAbs = d;
    err += d * d;
  }
  return {
    n, cos: dot / Math.sqrt(na * nb), max_abs: maxAbs,
    rel_l2: Math.sqrt(err / nb), snr_db: 10 * Math.log10(nb / Math.max(err, 1e-30)),
  };
}

async function fetchModel(url, key) {
  const t0 = now();
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  result.timings[`${key}_fetch_ms`] = now() - t0;
  result.sizes[`${key}_mb`] = bytes.byteLength / 2 ** 20;
  return bytes;
}

async function compile(bytes, key) {
  const t0 = now();
  const opts = { accelerator: ACCEL };
  if (THREADS) opts.cpuOptions = { numThreads: THREADS };
  const model = await loadAndCompile(bytes, opts);
  result.timings[`${key}_compile_ms`] = now() - t0;
  return model;
}

async function runSig(model, sig, inputs) {
  // inputs: Record<name, Tensor>; returns first output as a Float32Array copy.
  const out = await model.run(sig, inputs);
  const tensors = Object.values(out);
  const arr = tensors[0].toTypedArray().slice();
  for (const t of tensors) t.delete();
  return arr;
}

function padLen(src, ch, L, R) {
  if (L === R) return src;
  const out = new Float32Array(ch * R);
  for (let c = 0; c < ch; c++) out.set(src.subarray(c * L, c * L + L), c * R);
  return out;
}

function trimLen(src, ch, R, L) {
  if (L === R) return src;
  const out = new Float32Array(ch * L);
  for (let c = 0; c < ch; c++) out.set(src.subarray(c * R, c * R + L), c * L);
  return out;
}

// numpy float32 pingpong schedule (tflite_pipeline.build_pingpong_schedule, sigma_max = 1)
function pingpongSchedule(steps) {
  const f = Math.fround, out = new Float32Array(steps + 1);
  for (let i = 0; i <= steps; i++) {
    const t = f(1 - i / steps);
    const logsnr = f(2.0 - f(t * f(2.0 - -6.2)));
    out[i] = t <= 0 ? 0 : t >= 1 ? 1 : f(1 / f(1 + f(Math.exp(logsnr))));
  }
  out[0] = 1;
  return out;
}

// ---------- T5Gemma ----------
async function runT5(ids, mask) {
  const model = await compile(await fetchModel(MODELS.t5, "t5"), "t5");
  const det = [...model.getInputDetails()].sort((a, b) => (a.name < b.name ? -1 : 1));
  result.env.t5_inputs = det.map((d) => `${d.name}:${d.dtype}[${[...d.shape]}]`);
  const tIds = new Tensor(ids, [1, COND_TOKENS]);
  const tMask = new Tensor(mask, [1, COND_TOKENS]);
  const t0 = now();
  const out = await model.run({ [det[0].name]: tIds, [det[1].name]: tMask });
  const tensors = Object.values(out);
  const hidden = tensors[0].toTypedArray().slice();
  result.timings.t5_run_ms = now() - t0;
  for (const t of [tIds, tMask, ...tensors]) t.delete();
  model.delete();
  return hidden;
}

// ---------- T5Gemma, split (t5_split.py): embedding rows fetched by Range, body on LiteRT.js ----------
async function fetchEmbeddingRows(ids) {
  const D = COND_DIM, ROW = D * 2; // fp16
  const unique = [...new Set(ids)];
  const t0 = now();
  const rows = new Map(await Promise.all(unique.map(async (id) => {
    const r = await fetch("/sliced/t5_embed_fp16.bin", { headers: { Range: `bytes=${id * ROW}-${id * ROW + ROW - 1}` } });
    if (r.status !== 206) throw new Error(`range request for token ${id} returned ${r.status}`);
    return [id, new Float16Array(await r.arrayBuffer())];
  })));
  const embeds = new Float32Array(ids.length * D);
  ids.forEach((id, i) => embeds.set(rows.get(id), i * D)); // Float16 -> Float32 is exact
  result.timings.t5_rows_fetch_ms = now() - t0;
  result.sizes.t5_rows_kb = (unique.length * ROW) / 1024;
  result.env.t5_unique_tokens = unique.length;
  return embeds;
}

async function runT5Split(ids, mask) {
  const embeds = await fetchEmbeddingRows([...ids]);
  const model = await compile(await fetchModel("/sliced/t5_body_fp16.tflite", "t5"), "t5");
  const det = model.getInputDetails();
  result.env.t5_inputs = det.map((d) => `${d.name}:${d.dtype}[${[...d.shape]}]`);
  const tEmb = new Tensor(embeds, [1, COND_TOKENS, COND_DIM]);
  const tMask = new Tensor(mask, [1, COND_TOKENS]);
  const inputs = {};
  for (const d of det) inputs[d.name] = d.dtype === "float32" ? tEmb : tMask;
  const t0 = now();
  const out = await model.run(inputs);
  const tensors = Object.values(out);
  const hidden = tensors[0].toTypedArray().slice();
  result.timings.t5_run_ms = now() - t0;
  for (const t of [tEmb, tMask, ...tensors]) t.delete();
  model.delete();
  return hidden;
}

// ---------- DiT (rung ladder + gcond signature) ----------
class RungDiT {
  static async create(L, seconds) {
    const self = new RungDiT();
    self.model = await compile(await fetchModel(MODELS.dit, "dit"), "dit");
    const sigs = self.model.signatures;
    result.env.dit_signatures = Object.keys(sigs);
    self.sizes = Object.keys(sigs).filter((k) => /^s\d+$/.test(k)).map((k) => Number(k.slice(1))).sort((a, b) => a - b);
    self.L = L;
    self.R = self.sizes.find((s) => s >= L);
    self.sig = `s${self.R}`;
    self.seconds = seconds;
    const gdet = sigs.gcond.getInputDetails();
    result.env.gcond_inputs = gdet.map((d) => `${d.name}:${d.dtype}[${[...d.shape]}]`);
    self.gcNames = gdet.map((d) => d.name); // positional: [seconds, t] per rung_dit.py
    const det = sigs[self.sig].getInputDetails();
    result.env.dit_inputs = det.map((d) => `${d.name}:${d.dtype}[${[...d.shape]}]`);
    self.slots = {};
    for (const d of det) {
      const s = [...d.shape];
      if (s.length === 4) self.slots.am = d.name;
      else if (s.length === 2 && s[1] === COND_TOKENS) self.slots.t5m = d.name;
      else if (s.length === 2) { self.slots.gc = d.name; self.gceOut = s[1]; }
      else if (s.length === 3 && s[1] === 257) self.slots.lac = d.name;
      else if (s.length === 3 && s[2] === COND_DIM) self.slots.t5h = d.name;
      else if (s.length === 3 && s[1] === LATENT_CH) self.slots.x = d.name;
      else if (s.length === 1) self.slots.sec = d.name;
    }
    return self;
  }

  setConditioning(t5h, t5m) {
    const R = this.R;
    const am = new Float32Array(MEM + R);
    for (let i = MEM + this.L; i < MEM + R; i++) am[i] = -1e9;
    this.fixed = {
      t5h: [t5h, [1, COND_TOKENS, COND_DIM]],
      t5m: [t5m, [1, COND_TOKENS]],
      sec: [new Float32Array([this.seconds]), [1]],
      lac: [new Float32Array(257 * R), [1, 257, R]],
      am: [am, [1, 1, 1, MEM + R]],
    };
  }

  async gc(t) {
    const sec = new Tensor(new Float32Array([this.seconds]), [1]);
    const tt = new Tensor(new Float32Array([t]), [1]);
    const out = await runSig(this.model, "gcond", { [this.gcNames[0]]: sec, [this.gcNames[1]]: tt });
    sec.delete(); tt.delete();
    return out;
  }

  async forward(x, t) {
    const inputs = {}, owned = [];
    const add = (slot, data, shape) => { const ten = new Tensor(data, shape); owned.push(ten); inputs[this.slots[slot]] = ten; };
    for (const [slot, [data, shape]] of Object.entries(this.fixed)) add(slot, data, shape);
    add("x", padLen(x, LATENT_CH, this.L, this.R), [1, LATENT_CH, this.R]);
    add("gc", await this.gc(t), [1, this.gceOut]);
    const v = await runSig(this.model, this.sig, inputs);
    for (const ten of owned) ten.delete();
    return trimLen(v, LATENT_CH, this.R, this.L);
  }
}

// rectified-flow pingpong sampler (tflite_pipeline.sample), float32 arithmetic like numpy
async function sample(dit, x0, noise, sigmas, onStep) {
  const f = Math.fround, steps = sigmas.length - 1, n = x0.length;
  let x = x0.slice();
  for (let i = 0; i < steps; i++) {
    const tc = f(sigmas[i]), tn = f(sigmas[i + 1]);
    const v = await dit.forward(x, tc);
    const next = new Float32Array(n);
    const renoise = i < steps - 1 && tn > 0;
    const nz = noise.subarray(i * n, (i + 1) * n);
    for (let j = 0; j < n; j++) {
      const den = f(x[j] - f(tc * v[j]));
      next[j] = renoise ? f(f(f(1 - tn) * den) + f(tn * nz[j])) : den;
    }
    x = next;
    onStep?.(i);
  }
  return x;
}

// ---------- SAME-S rung decoder (rung_decoder.RungDecoder.decode, balance=True) ----------
class RungDecoder {
  static async create(trim) {
    const self = new RungDecoder();
    self.model = await compile(await fetchModel(MODELS.dec, "dec"), "dec");
    self.sizes = Object.keys(self.model.signatures).filter((k) => /^s\d+$/.test(k)).map((k) => Number(k.slice(1))).sort((a, b) => a - b);
    result.env.dec_rungs = self.sizes;
    self.trim = trim;
    return self;
  }
  tileable(R) { return R - 2 * this.trim > 0; }
  nwin(L, C) {
    if (L <= C) return 1;
    const step = C - 2 * this.trim;
    if (step <= 0) return 1e9;
    const starts = [];
    for (let s = 0; s < L - C; s += step) starts.push(s);
    if (!starts.length || starts.at(-1) !== L - C) starts.push(L - C);
    return starts.length;
  }
  bestRung(L) {
    const cands = [];
    for (const R of this.sizes) {
      if (R === L) cands.push([R, 1, R]);
      else if (R < L && this.tileable(R)) { const nw = this.nwin(L, R); cands.push([nw * R, nw, R]); }
    }
    if (!cands.length) return this.sizes.find((R) => R >= L) ?? this.sizes.at(-1);
    cands.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    return cands[0][2];
  }
  async runRung(lat, L, start, size) {
    // lat: [256, Ltot] channel-major; window [start, start+min(size, L-start))
    const n = Math.min(size, L - start), win = new Float32Array(LATENT_CH * size);
    for (let c = 0; c < LATENT_CH; c++) {
      const row = lat.subarray(c * L + start, c * L + start + n);
      win.set(row, c * size);
      for (let k = n; k < size; k++) win[c * size + k] = row[n - 1];
    }
    const ten = new Tensor(win, [1, LATENT_CH, size]);
    const name = this.model.signatures[`s${size}`].getInputDetails()[0].name;
    const y = await runSig(this.model, `s${size}`, { [name]: ten });
    ten.delete();
    return y; // [2, size*SPL]
  }
  async decode(lat, L) {
    const C = this.bestRung(L);
    this.lastRung = C;
    const S = SPL, out = new Float32Array(2 * L * S);
    const put = (y, dstStart, srcLo, srcHi) => {
      for (let ch = 0; ch < 2; ch++) out.set(y.subarray(ch * C * S + srcLo * S, ch * C * S + srcHi * S), ch * L * S + dstStart * S);
    };
    if (L <= C) { const y = await this.runRung(lat, L, 0, C); put(y, 0, 0, L); return out; }
    const step = C - 2 * this.trim, starts = [];
    for (let s = 0; s < L - C; s += step) starts.push(s);
    if (!starts.length || starts.at(-1) !== L - C) starts.push(L - C);
    for (let k = 0; k < starts.length; k++) {
      const st = starts[k];
      const y = await this.runRung(lat, L, st, C);
      const lo = k === 0 ? 0 : STITCH_TRIM;
      const hi = st + C >= L ? C : C - STITCH_TRIM;
      put(y, st + lo, lo, hi);
    }
    return out;
  }
}

function padOddLatent(lat, L) {
  // SAME-S needs even L: repeat the last latent (sa3_tflite.py)
  const out = new Float32Array(LATENT_CH * (L + 1));
  for (let c = 0; c < LATENT_CH; c++) {
    out.set(lat.subarray(c * L, c * L + L), c * (L + 1));
    out[c * (L + 1) + L] = lat[c * L + L - 1];
  }
  return out;
}

async function decodeAudio(dec, latents, T_lat, nSamples) {
  let lat = latents, L = T_lat;
  if (L % 2) { lat = padOddLatent(latents, L); L += 1; }
  const t0 = now();
  const full = await dec.decode(lat, L);
  const ms = now() - t0;
  const audio = new Float32Array(2 * nSamples);
  for (let ch = 0; ch < 2; ch++) audio.set(full.subarray(ch * L * SPL, ch * L * SPL + nSamples), ch * nSamples);
  return { audio, ms };
}

async function upload(name, data) {
  await fetch(`/upload/${name}`, { method: "POST", body: data });
}

// ---------- main ----------
async function main() {
  result.env.userAgent = navigator.userAgent;
  result.env.crossOriginIsolated = self.crossOriginIsolated;
  result.env.hardwareConcurrency = navigator.hardwareConcurrency;
  result.env.threads = THREADS;
  result.env.accel = ACCEL;
  result.env.sliced = SLICED;
  const manifest = await (await fetch("/parity/manifest.json")).json();
  result.env.native = { timings: manifest.timings, threads: manifest.threads, T_lat: manifest.T_lat, prompt: manifest.prompt };
  const A = {};
  for (const k of Object.keys(manifest.arrays)) A[k] = await loadArray(manifest, k);
  const T_lat = manifest.T_lat;
  const nSamples = A.audio.shape[1];

  let t0 = now();
  await loadLiteRt("/litert/wasm/", THREADS ? { threads: true } : {});
  result.timings.litert_load_ms = now() - t0;

  // schedule
  const sigmas = pingpongSchedule(manifest.steps);
  result.parity.sigmas = compare(sigmas, A.sigmas.data);

  // 1) T5
  let t5h = A.t5_hidden.data;
  if (STAGES.has("t5")) {
    log("T5Gemma ...");
    t5h = SLICED ? await runT5Split(A.ids.data, A.mask.data) : await runT5(A.ids.data, A.mask.data);
    result.parity.t5_hidden = compare(t5h, A.t5_hidden.data);
    log("t5", JSON.stringify(result.parity.t5_hidden));
  }
  const t5m = Float32Array.from(A.mask.data);

  // 2) DiT
  let latents = null;
  if (STAGES.has("gcond") || STAGES.has("dit1") || STAGES.has("sample")) {
    log("DiT ...");
    const dit = await RungDiT.create(T_lat, manifest.seconds);
    result.env.dit_rung = dit.R;
    if (STAGES.has("gcond")) {
      result.parity.gc_step1 = compare(await dit.gc(sigmas[0]), A.gc_step1.data);
      log("gc", JSON.stringify(result.parity.gc_step1));
    }
    if (STAGES.has("dit1")) {
      dit.setConditioning(A.t5_hidden.data, t5m); // native T5 output: isolates one DiT forward
      t0 = now();
      const v1 = await dit.forward(A.x0.data, sigmas[0]);
      result.timings.dit_first_forward_ms = now() - t0;
      result.parity.v_step1 = compare(v1, A.v_step1.data);
      if (A.v_step1_fp32) {
        // quality vs the fp32 DiT: is the browser's int8 worse than native int8, or just different?
        result.parity.v_step1_browser_vs_fp32 = compare(v1, A.v_step1_fp32.data);
        result.parity.v_step1_native_int8_vs_fp32 = compare(A.v_step1.data, A.v_step1_fp32.data);
      }
      log("v1", JSON.stringify(result.parity.v_step1));
    }
    if (STAGES.has("sample")) {
      dit.setConditioning(t5h, t5m); // browser T5 output: the end-to-end path
      const stepMs = [];
      let last = now();
      t0 = last;
      latents = await sample(dit, A.x0.data, A.step_noise.data, sigmas, () => { const n = now(); stepMs.push(n - last); last = n; });
      result.timings.dit_sample_ms = now() - t0;
      result.timings.dit_step_ms = stepMs;
      result.parity.latents = compare(latents, A.latents.data);
      log("latents", JSON.stringify(result.parity.latents));
    }
    dit.model.delete();
  }

  // 3) Decoder
  if (STAGES.has("decode_native") || STAGES.has("decode")) {
    log("decoder ...");
    const dec = await RungDecoder.create(manifest.trim);
    if (STAGES.has("decode_native")) {
      const { audio, ms } = await decodeAudio(dec, A.latents.data, T_lat, nSamples);
      result.timings.dec_native_latents_ms = ms;
      result.env.dec_rung = dec.lastRung;
      result.parity.audio_from_native_latents = compare(audio, A.audio.data);
      log("audio(native latents)", JSON.stringify(result.parity.audio_from_native_latents));
    }
    if (STAGES.has("decode") && latents) {
      const { audio, ms } = await decodeAudio(dec, latents, T_lat, nSamples);
      result.timings.dec_run_ms = ms;
      result.parity.audio_end_to_end = compare(audio, A.audio.data);
      let peak = 0, ss = 0;
      for (const v of audio) { peak = Math.max(peak, Math.abs(v)); ss += v * v; }
      result.env.browser_audio = { peak, rms: Math.sqrt(ss / audio.length), samples_per_channel: nSamples };
      await upload(`browser_audio_${THREADS ? `t${THREADS}` : "st"}.f32`, audio);
    }
    dec.model.delete();
  }
}

main()
  .catch((e) => { result.errors.push(String(e?.stack || e)); console.error(e); })
  .finally(() => { window.__spike.done = true; log("done"); });
