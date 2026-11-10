// Drive the browser spike in headless Chromium and record results.
//   node run_spike.mjs [--threads N] [--sliced 1] [--dit URL] [--stages a,b] [--label NAME] [--port 8765]
// Expects serve.mjs to be running. Samples the renderer process RSS every 200 ms (WASM heap
// lives there) and writes out/result_<label>.json, plus out/browser_<label>.wav if the page
// uploaded audio.
import { execSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
const HERE = fileURLToPath(new URL(".", import.meta.url));

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
const threads = Number(args.threads || 0);
const label = args.label || (threads ? `t${threads}` : "st");
const port = Number(args.port || 8765);
const qs = new URLSearchParams({
  threads: String(threads),
  ...(args.stages ? { stages: args.stages } : {}),
  ...(args.sliced ? { sliced: args.sliced } : {}),
  ...(args.dit ? { dit: args.dit } : {}),
});

function rendererRssMb() {
  try {
    const out = execSync("ps -eo rss,args", { encoding: "utf8" });
    return out.split("\n").filter((l) => l.includes("--type=renderer"))
      .reduce((s, l) => s + Number(l.trim().split(/\s+/)[0]) / 1024, 0);
  } catch { return 0; }
}

const browser = await chromium.launch();
const page = await browser.newPage();
page.on("console", (m) => { if (m.text().startsWith("[spike]") || m.type() === "error" || m.type() === "warning") console.log(`[page:${m.type()}] ${m.text()}`); });
page.on("pageerror", (e) => console.log(`[pageerror] ${e}`));

let peak = 0;
const sampler = setInterval(() => { peak = Math.max(peak, rendererRssMb()); }, 200);
const t0 = Date.now();
await page.goto(`http://127.0.0.1:${port}/?${qs}`);
await page.waitForFunction(() => window.__spike?.done, null, { timeout: 40 * 60 * 1000, polling: 500 });
clearInterval(sampler);
const result = await page.evaluate(() => window.__spike.result);
result.env.wall_ms = Date.now() - t0;
result.env.peak_renderer_rss_mb = peak;
await browser.close();

const resultPath = join(HERE, "out", `result_${label}.json`);
writeFileSync(resultPath, JSON.stringify(result, null, 2));

// float32 planar [2, n] -> 16-bit PCM WAV (same quantiser as tflite_pipeline.save_wav)
const f32Path = join(HERE, "out", `browser_audio_${threads ? `t${threads}` : "st"}.f32`);
if (existsSync(f32Path)) {
  const buf = readFileSync(f32Path);
  const f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const n = f.length / 2, sr = 44100, data = Buffer.alloc(n * 4), hdr = Buffer.alloc(44);
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
    const v = Math.max(-1, Math.min(1, f[c * n + i]));
    data.writeInt16LE(Math.round(v * 32767), (i * 2 + c) * 2);
  }
  hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + data.length, 4); hdr.write("WAVE", 8); hdr.write("fmt ", 12);
  hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(2, 22); hdr.writeUInt32LE(sr, 24);
  hdr.writeUInt32LE(sr * 4, 28); hdr.writeUInt16LE(4, 32); hdr.writeUInt16LE(16, 34); hdr.write("data", 36);
  hdr.writeUInt32LE(data.length, 40);
  writeFileSync(join(HERE, "out", `browser_${label}.wav`), Buffer.concat([hdr, data]));
}
console.log(JSON.stringify({ label, errors: result.errors, timings: result.timings, parity: result.parity, peak_renderer_rss_mb: peak }, null, 2));
