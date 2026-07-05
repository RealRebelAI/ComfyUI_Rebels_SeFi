#!/usr/bin/env python
"""
sefi_merge_transformer.py
=========================
Merge SeFi-Image sharded transformer safetensors into ONE .safetensors file,
fully streaming (one tensor in RAM at a time, <1GB peak) -- safe on 16GB RAM.

Writes the safetensors format manually (8-byte header length + JSON header +
raw tensor buffer) so the whole model never has to sit in memory, unlike
safetensors.torch.save_file.

USAGE:
  python_embeded\\python.exe D:\\sefi_merge_transformer.py ^
     --src D:\\sefi-image\\transformer ^
     --dst D:\\sefi-image\\SeFi-Image-5B-Base_transformer_bf16.safetensors

--src points at the transformer FOLDER containing the shards + index.json.
"""
import argparse
import json
import os
import struct

from safetensors import safe_open

DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

# torch dtype string -> safetensors dtype tag
TORCH_TO_ST = {
    "torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32",
    "torch.float64": "F64", "torch.int8": "I8", "torch.uint8": "U8",
    "torch.int16": "I16", "torch.int32": "I32", "torch.int64": "I64",
    "torch.bool": "BOOL",
    "torch.float8_e4m3fn": "F8_E4M3", "torch.float8_e5m2": "F8_E5M2",
}


def load_shard_map(src_dir):
    idx = os.path.join(src_dir, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]  # key -> shard file
        shards = {}
        for key, fn in weight_map.items():
            shards.setdefault(fn, []).append(key)
        return shards
    # single-file or unindexed fallback
    shards = {}
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(src_dir, fn), framework="pt", device="cpu") as f:
                shards[fn] = list(f.keys())
    if not shards:
        raise FileNotFoundError(f"No .safetensors shards found in {src_dir}")
    return shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="transformer folder with shards + index.json")
    ap.add_argument("--dst", required=True, help="output single .safetensors path")
    a = ap.parse_args()

    shards = load_shard_map(a.src)

    # ---- pass 1: metadata only (shapes/dtypes via slices, no tensor loads) ----
    entries = []  # (shard_file, key, st_dtype, shape, nbytes)
    for fn in sorted(shards):
        path = os.path.join(a.src, fn)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in shards[fn]:
                sl = f.get_slice(key)
                shape = list(sl.get_shape())
                tdt = str(sl.get_dtype())
                # safe_open returns e.g. "torch.bfloat16" or "BF16" depending on version
                st_dtype = TORCH_TO_ST.get(tdt, tdt.upper().replace("TORCH.", ""))
                if st_dtype not in DTYPE_BYTES:
                    raise ValueError(f"Unhandled dtype {tdt} for tensor {key}")
                n = 1
                for d in shape:
                    n *= d
                entries.append((fn, key, st_dtype, shape, n * DTYPE_BYTES[st_dtype]))

    # stable key order = sorted by name (safetensors convention-friendly)
    entries.sort(key=lambda e: e[1])

    header = {}
    offset = 0
    for _, key, st_dtype, shape, nbytes in entries:
        header[key] = {
            "dtype": st_dtype,
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header["__metadata__"] = {"format": "pt", "merged_by": "sefi_merge_transformer.py"}

    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8  # 8-byte align like the reference impl
    header_bytes += b" " * pad

    total_gb = offset / 1e9
    print(f"tensors: {len(entries)} | payload: {total_gb:.2f} GB | writing {a.dst}")

    # ---- pass 2: stream tensor bytes shard by shard, in header order ----
    with open(a.dst, "wb") as out:
        out.write(struct.pack("<Q", len(header_bytes)))
        out.write(header_bytes)

        open_shard = {"fn": None, "f": None, "ctx": None}

        def get_handle(fn):
            if open_shard["fn"] != fn:
                if open_shard["ctx"] is not None:
                    open_shard["ctx"].__exit__(None, None, None)
                ctx = safe_open(os.path.join(a.src, fn), framework="pt", device="cpu")
                open_shard.update(fn=fn, ctx=ctx, f=ctx.__enter__())
            return open_shard["f"]

        import torch

        def tensor_bytes(t):
            """Raw little-endian bytes for any dtype, incl. bf16/fp8 (no numpy dtype)."""
            t = t.contiguous()
            if t.dtype == torch.bfloat16:
                return t.view(torch.int16).numpy().tobytes()
            if t.dtype in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
            ):
                return t.view(torch.int8).numpy().tobytes()
            return t.numpy().tobytes()

        written = 0
        for i, (fn, key, st_dtype, shape, nbytes) in enumerate(entries, 1):
            f = get_handle(fn)
            t = f.get_tensor(key)
            out.write(tensor_bytes(t))
            written += nbytes
            del t
            if i % 100 == 0:
                print(f"  [{i}/{len(entries)}] {written/1e9:.2f} GB written", flush=True)

        if open_shard["ctx"] is not None:
            open_shard["ctx"].__exit__(None, None, None)

    final = os.path.getsize(a.dst)
    print(f"DONE -> {a.dst} ({final/1e9:.2f} GB)")
    if final != 8 + len(header_bytes) + offset:
        raise RuntimeError("Size mismatch - merged file may be corrupt, do not use.")
    # verify readable
    with safe_open(a.dst, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    print(f"verify: reopened OK, {len(keys)} tensors")


if __name__ == "__main__":
    main()
