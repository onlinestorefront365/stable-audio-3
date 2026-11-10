// Static server for the browser spike. Sends the cross-origin isolation headers that
// SharedArrayBuffer (and so LiteRT.js's multithreaded WASM build) requires, and accepts
// result uploads from the page.
//
//   /                  -> this directory (index.html, spike.js)
//   /litert/...        -> node_modules/@litertjs/core (dist + wasm)
//   /wasm-utils/...    -> node_modules/@litertjs/wasm-utils
//   /models/...        -> ../tflite/models/tflite (the native runtime's downloaded .tflite files)
//   /parity/...        -> ./parity (native_ref.py dumps)
//   /sliced/...        -> ./sliced (slice_signatures.py output)
//   POST /upload/NAME  -> ./out/NAME
import { createReadStream, createWriteStream, mkdirSync, realpathSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const ROOTS = {
  "/litert/": join(HERE, "node_modules/@litertjs/core/"),
  "/wasm-utils/": join(HERE, "node_modules/@litertjs/wasm-utils/"),
  "/models/": resolve(HERE, "../tflite/models/tflite/"),
  "/parity/": join(HERE, "parity/"),
  "/sliced/": join(HERE, "sliced/"),
  "/": HERE,
};
const TYPES = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json", ".map": "application/json",
};
const OUT = join(HERE, "out");
mkdirSync(OUT, { recursive: true });

const isolate = (res) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  res.setHeader("Cache-Control", "no-store");
};

const server = createServer((req, res) => {
  isolate(res);
  const url = new URL(req.url, "http://x");
  const path = decodeURIComponent(url.pathname);

  if (req.method === "POST" && path.startsWith("/upload/")) {
    const name = path.slice("/upload/".length).replace(/[^\w.-]/g, "_");
    const ws = createWriteStream(join(OUT, name));
    req.pipe(ws);
    ws.on("finish", () => { res.writeHead(200); res.end("ok"); });
    return;
  }

  const prefix = Object.keys(ROOTS).find((p) => path.startsWith(p));
  const base = ROOTS[prefix];
  const target = normalize(join(base, path.slice(prefix.length) || "index.html"));
  if (!target.startsWith(base)) { res.writeHead(403); res.end(); return; }
  let real;
  try {
    real = realpathSync(target); // model files are symlinks into the HF cache
    if (!statSync(real).isFile()) throw new Error("not a file");
  } catch {
    res.writeHead(404); res.end(); return;
  }
  const size = statSync(real).size;
  const type = TYPES[extname(target)] || "application/octet-stream";
  const range = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range || "");
  if (range) { // single-range requests: the page fetches individual T5 embedding rows
    const start = Number(range[1]), end = Math.min(Number(range[2]), size - 1);
    res.writeHead(206, { "Content-Type": type, "Content-Length": end - start + 1, "Content-Range": `bytes ${start}-${end}/${size}` });
    createReadStream(real, { start, end }).pipe(res);
    return;
  }
  res.writeHead(200, { "Content-Type": type, "Content-Length": size, "Accept-Ranges": "bytes" });
  createReadStream(real).pipe(res);
});

const port = Number(process.env.PORT || 8765);
server.listen(port, "127.0.0.1", () => console.log(`spike server on http://127.0.0.1:${port}`));
