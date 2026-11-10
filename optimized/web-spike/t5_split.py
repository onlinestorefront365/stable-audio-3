"""Split T5Gemma's token-embedding lookup out of encoder_fp16.tflite.

The encoder's first ops are  ids -> RESHAPE -> EMBEDDING_LOOKUP(DEQUANTIZE(fp16 table)) -> RESHAPE.
The table is 256000 x 768 (197M of the encoder's 282M weights), and DEQUANTIZE expands all of it
to fp32 (786 MB) at runtime, which together with the rest overflows LiteRT.js's 2 GB WASM heap.

This writes:
  t5_body_fp16.tflite   the encoder with those four ops removed; its first input is now the
                        looked-up embeddings, float32 [1, 256, 768] ("embeds"); the mask input
                        is unchanged
  t5_embed_fp16.bin     the raw fp16 table, row-major [256000, 768]. A browser fetches only the
                        rows it needs (HTTP Range requests, 1536 bytes per token), so the table
                        is never downloaded whole.

    python t5_split.py SRC OUT_DIR
"""

import sys
from pathlib import Path

import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema

OPS = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}


def main(src, out_dir):
    out_dir = Path(out_dir)
    raw = bytearray(open(src, "rb").read())
    mt = schema.ModelT.InitFromObj(schema.Model.GetRootAs(raw, 0))
    sg = mt.subgraphs[0]

    def opname(op):
        oc = mt.operatorCodes[op.opcodeIndex]
        return OPS[max(oc.builtinCode, oc.deprecatedBuiltinCode)]

    producer = {t: i for i, op in enumerate(sg.operators) for t in op.outputs}
    consumers = {}
    for i, op in enumerate(sg.operators):
        for t in op.inputs:
            consumers.setdefault(t, []).append(i)

    (lookup,) = [i for i, op in enumerate(sg.operators) if opname(op) == "EMBEDDING_LOOKUP"]
    ids_t, table_t = sg.operators[lookup].inputs
    deq = producer[table_t]
    assert opname(sg.operators[deq]) == "DEQUANTIZE"
    fp16_t = sg.operators[deq].inputs[0]
    pre = producer[ids_t]
    assert opname(sg.operators[pre]) == "RESHAPE" and sg.operators[pre].inputs[0] == sg.inputs[0]
    (post,) = consumers[sg.operators[lookup].outputs[0]]
    assert opname(sg.operators[post]) == "RESHAPE"
    embeds_t = sg.operators[post].outputs[0]
    assert list(sg.tensors[embeds_t].shape) == [1, 256, 768], sg.tensors[embeds_t].shape

    # The exporter stores large weights out-of-band (Buffer.offset/size into the file). Pull every
    # buffer in-band (without the table the body is ~145 MB, far below the 2 GB flatbuffer limit).
    for b in mt.buffers:
        if b.offset and b.offset > 1:
            b.data = np.frombuffer(raw, dtype=np.uint8, count=b.size, offset=b.offset).copy()
            b.offset = b.size = 0

    # export the fp16 table, then drop its buffer from the model
    fp16 = sg.tensors[fp16_t]
    table = np.frombuffer(mt.buffers[fp16.buffer].data.tobytes(), dtype="<f2").reshape(list(fp16.shape))
    table.tofile(out_dir / "t5_embed_fp16.bin")
    mt.buffers[fp16.buffer].data = None

    old_ids_input = sg.inputs[0]
    sg.operators = [op for i, op in enumerate(sg.operators) if i not in {pre, deq, lookup, post}]
    sg.inputs = [embeds_t] + list(sg.inputs[1:])
    sg.tensors[embeds_t].name = b"embeds"
    for sig in mt.signatureDefs:
        for tm in sig.inputs:
            if tm.tensorIndex == old_ids_input:
                tm.tensorIndex, tm.name = embeds_t, b"embeds"

    builder = flatbuffers.Builder(1 << 20)
    builder.Finish(mt.Pack(builder), file_identifier=b"TFL3")
    (out_dir / "t5_body_fp16.tflite").write_bytes(builder.Output())
    print(f"table {table.shape} -> t5_embed_fp16.bin; body ops {len(sg.operators)}; inputs {sg.inputs}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
