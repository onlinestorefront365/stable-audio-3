"""Cut a multi-signature rung .tflite down to the signatures one generation needs.

LiteRT.js prepares every signature at compile time and, with no XNNPACK weight cache in the
browser, packs the shared weights once per signature. For the 8-rung small DiT that overflows
the 2 GB WASM heap, so the browser gets a file with only (e.g.) 'gcond' + 's416'.

Only in-band models (all buffers inside the flatbuffer, true for the w8a8 DiT and the SAME
codecs) are supported. Unreferenced buffers are dropped and tensor buffer indices remapped.

    python slice_signatures.py SRC DST KEY [KEY ...]
"""

import sys

import flatbuffers
from ai_edge_litert import schema_py_generated as schema


def _key(sig):
    k = sig.signatureKey
    return k.decode() if isinstance(k, bytes) else k


def slice_model(src, dst, keep):
    raw = bytearray(open(src, "rb").read())
    model = schema.Model.GetRootAs(raw, 0)
    if any(model.Buffers(i).Offset() > 1 for i in range(model.BuffersLength())):
        raise SystemExit(f"{src}: out-of-band buffers are not supported")
    mt = schema.ModelT.InitFromObj(model)

    sigs = [s for s in mt.signatureDefs if _key(s) in keep]
    missing = set(keep) - {_key(s) for s in sigs}
    if missing:
        raise SystemExit(f"{src}: no signature(s) {sorted(missing)}; have {[_key(s) for s in mt.signatureDefs]}")
    mt.subgraphs = [mt.subgraphs[s.subgraphIndex] for s in sigs]
    for new_index, s in enumerate(sigs):
        s.subgraphIndex = new_index
    mt.signatureDefs = sigs

    used = {0}  # buffer 0 is the empty sentinel
    for sg in mt.subgraphs:
        used.update(t.buffer for t in sg.tensors)
    used.update(md.buffer for md in (mt.metadata or []))
    used.update(mt.metadataBuffer or [])
    order = sorted(used)
    remap = {old: new for new, old in enumerate(order)}
    mt.buffers = [mt.buffers[i] for i in order]
    for sg in mt.subgraphs:
        for t in sg.tensors:
            t.buffer = remap[t.buffer]
    for md in mt.metadata or []:
        md.buffer = remap[md.buffer]
    if mt.metadataBuffer:
        mt.metadataBuffer = [remap[b] for b in mt.metadataBuffer]

    builder = flatbuffers.Builder(1 << 20)
    builder.Finish(mt.Pack(builder), file_identifier=b"TFL3")
    with open(dst, "wb") as f:
        f.write(builder.Output())
    print(f"{dst}: signatures {[_key(s) for s in sigs]}, buffers {len(order)}/{model.BuffersLength()}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    slice_model(sys.argv[1], sys.argv[2], sys.argv[3:])
