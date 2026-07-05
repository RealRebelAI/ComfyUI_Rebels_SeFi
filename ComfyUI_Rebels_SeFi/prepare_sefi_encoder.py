#!/usr/bin/env python
r"""
prepare_sefi_encoder.py  --  run ONCE, then everything is dropdown-driven.

Does two things:
  1. Downloads the SMALL Qwen3-VL-4B assets (config.json, tokenizer, processor
     configs -- a few MB, Apache-2.0) into this node pack's encoder_assets/
     folder, so the pack is self-contained for every user.
  2. Streams the encoder weights into ONE .safetensors (visual tower STRIPPED --
     SeFi only uses the text stack, saves ~1GB) and drops it into
     ComfyUI/models/text_encoders/ for the loader dropdown.

USAGE (from anywhere):
  python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_Rebels_SeFi\prepare_sefi_encoder.py ^
      --comfy D:\AI_Tools\ComfyUI_windows_portable\ComfyUI

Optional: --src <local Qwen3-VL-4B-Instruct folder> to skip the download and
merge from a folder you already have.
"""
import argparse
import json
import os
import struct
import sys

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACK_DIR, "encoder_assets", "qwen3vl_4b")
REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
SMALL_ASSETS = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "chat_template.json",
    "special_tokens_map.json",
]
OUT_NAME = "SeFi_Qwen3-VL-4B_text_bf16.safetensors"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1,
               "I16": 2, "I32": 4, "I64": 8, "BOOL": 1}
TORCH_TO_ST = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32",
               "torch.float64": "F64", "torch.int8": "I8", "torch.uint8": "U8",
               "torch.int16": "I16", "torch.int32": "I32", "torch.int64": "I64",
               "torch.bool": "BOOL"}


def fetch_assets(src_dir):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    if src_dir:
        import shutil
        copied = 0
        for name in SMALL_ASSETS:
            p = os.path.join(src_dir, name)
            if os.path.isfile(p):
                shutil.copy2(p, os.path.join(ASSETS_DIR, name))
                copied += 1
        print(f"assets: copied {copied} files from {src_dir} -> {ASSETS_DIR}")
        return src_dir
    from huggingface_hub import hf_hub_download, snapshot_download
    got = 0
    for name in SMALL_ASSETS:
        try:
            hf_hub_download(REPO_ID, name, local_dir=ASSETS_DIR)
            got += 1
        except Exception:
            pass  # not every repo ships every file
    if got == 0:
        sys.exit("Could not download any encoder assets - check network / repo id.")
    print(f"assets: downloaded {got} files -> {ASSETS_DIR}")
    print("downloading encoder weights (sharded, ~9GB) ...")
    return snapshot_download(REPO_ID, allow_patterns=["*.safetensors", "*.safetensors.index.json"])


def stream_merge(src_dir, dst_path):
    """Merge shards -> one file, dropping the visual tower. <1GB RAM."""
    import torch
    from safetensors import safe_open

    idx = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.isfile(idx):
        with open(idx, "r", encoding="utf-8") as f:
            wm = json.load(f)["weight_map"]
        shards = {}
        for k, fn in wm.items():
            shards.setdefault(fn, []).append(k)
    else:
        shards = {}
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(src_dir, fn), framework="pt", device="cpu") as f:
                    shards[fn] = list(f.keys())
    if not shards:
        sys.exit(f"No safetensors shards found in {src_dir}")

    def keep(key):
        return ".visual." not in key and not key.startswith("visual.")

    entries = []
    for fn in sorted(shards):
        with safe_open(os.path.join(src_dir, fn), framework="pt", device="cpu") as f:
            for k in shards[fn]:
                if not keep(k):
                    continue
                sl = f.get_slice(k)
                shape = list(sl.get_shape())
                dt = TORCH_TO_ST.get(str(sl.get_dtype()), str(sl.get_dtype()).upper().replace("TORCH.", ""))
                if dt not in DTYPE_BYTES:
                    sys.exit(f"Unhandled dtype {dt} for {k}")
                n = 1
                for d in shape:
                    n *= d
                entries.append((fn, k, dt, shape, n * DTYPE_BYTES[dt]))
    entries.sort(key=lambda e: e[1])

    header, off = {}, 0
    for _, k, dt, shape, nb in entries:
        header[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    header["__metadata__"] = {"format": "pt", "note": "visual tower stripped for SeFi text encoding"}
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    print(f"merging {len(entries)} text tensors ({off/1e9:.2f} GB) -> {dst_path}")

    def tbytes(t):
        t = t.contiguous()
        if t.dtype == torch.bfloat16:
            return t.view(torch.int16).numpy().tobytes()
        return t.numpy().tobytes()

    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        cur = {"fn": None, "ctx": None, "f": None}
        for i, (fn, k, dt, shape, nb) in enumerate(entries, 1):
            if cur["fn"] != fn:
                if cur["ctx"]:
                    cur["ctx"].__exit__(None, None, None)
                ctx = safe_open(os.path.join(src_dir, fn), framework="pt", device="cpu")
                cur.update(fn=fn, ctx=ctx, f=ctx.__enter__())
            out.write(tbytes(cur["f"].get_tensor(k)))
            if i % 100 == 0:
                print(f"  [{i}/{len(entries)}]", flush=True)
        if cur["ctx"]:
            cur["ctx"].__exit__(None, None, None)

    if os.path.getsize(dst_path) != 8 + len(hb) + off:
        sys.exit("Size mismatch - merged encoder corrupt, do not use.")
    with safe_open(dst_path, framework="pt", device="cpu") as f:
        n = len(list(f.keys()))
    print(f"DONE -> {dst_path} ({os.path.getsize(dst_path)/1e9:.2f} GB, {n} tensors, verify OK)")


def convert_from_fp8(fp8_path, dst_path):
    """Build the merged bf16 text encoder from a ComfyUI scaled-fp8 file
    (e.g. Comfy-Org qwen3vl_4b_fp8_scaled.safetensors). Streaming, low RAM.

    Key map: model.X -> model.language_model.X ; visual tower + comfy_quant
    markers dropped ; fp8 weights dequantized as weight * weight_scale.
    """
    import torch
    from safetensors import safe_open

    with safe_open(fp8_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

    def is_text(k):
        return not k.startswith("model.visual.") and not k.endswith(".comfy_quant")

    def remap(k):
        if k.startswith("model."):
            return "model.language_model." + k[len("model."):]
        return k

    scales = {k for k in all_keys if k.endswith(".weight_scale")}
    out_entries = []  # (src_key, out_key, shape) - dequant means all bf16
    with safe_open(fp8_path, framework="pt", device="cpu") as f:
        for k in all_keys:
            if not is_text(k) or k in scales:
                continue
            shape = list(f.get_slice(k).get_shape())
            out_entries.append((k, remap(k), shape))
    out_entries.sort(key=lambda e: e[1])

    header, off = {}, 0
    for _, ok, shape in out_entries:
        n = 1
        for d in shape:
            n *= d
        header[ok] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + n * 2]}
        off += n * 2
    header["__metadata__"] = {"format": "pt", "note": "dequantized from ComfyUI scaled-fp8; visual tower stripped"}
    hb = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    print(f"converting {len(out_entries)} text tensors from fp8 ({off/1e9:.2f} GB out) -> {dst_path}")

    with open(dst_path, "wb") as out, safe_open(fp8_path, framework="pt", device="cpu") as f:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for i, (sk, ok, shape) in enumerate(out_entries, 1):
            t = f.get_tensor(sk)
            sc_key = sk + "_scale" if sk + "_scale" in scales else (
                sk.rsplit(".", 1)[0] + ".weight_scale" if sk.endswith(".weight") else None)
            if t.dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
                scale = f.get_tensor(sc_key).to(torch.float32) if (sc_key and sc_key in scales) else torch.tensor(1.0)
                w = t.to(torch.float32)
                if scale.ndim == 1 and w.ndim == 2 and scale.shape[0] == w.shape[0]:
                    scale = scale.view(-1, 1)
                t = (w * scale).to(torch.bfloat16)
            else:
                t = t.to(torch.bfloat16)
            out.write(t.contiguous().view(torch.int16).numpy().tobytes())
            del t
            if i % 100 == 0:
                print(f"  [{i}/{len(out_entries)}]", flush=True)

    if os.path.getsize(dst_path) != 8 + len(hb) + off:
        sys.exit("Size mismatch - converted encoder corrupt, do not use.")
    from safetensors import safe_open as so
    with so(dst_path, framework="pt", device="cpu") as f:
        n = len(list(f.keys()))
    print(f"DONE -> {dst_path} ({os.path.getsize(dst_path)/1e9:.2f} GB, {n} tensors, verify OK)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True, help="path to your ComfyUI folder (the one containing models/)")
    ap.add_argument("--src", default=None, help="optional local Qwen3-VL-4B-Instruct folder (skips download)")
    ap.add_argument("--from-fp8", default=None, dest="from_fp8",
                    help="build from a ComfyUI scaled-fp8 encoder file (e.g. qwen3vl_4b_fp8_scaled.safetensors); skips the 9GB weight download (small config/tokenizer assets still fetched)")
    a = ap.parse_args()

    te_dir = os.path.join(a.comfy, "models", "text_encoders")
    if not os.path.isdir(te_dir):
        sys.exit(f"Not a ComfyUI folder (no models/text_encoders): {a.comfy}")

    if a.from_fp8:
        # small assets only (config/tokenizer); weights come from the fp8 file
        if a.src:
            fetch_assets(a.src)
        else:
            from huggingface_hub import hf_hub_download
            os.makedirs(ASSETS_DIR, exist_ok=True)
            got = 0
            for name in SMALL_ASSETS:
                try:
                    hf_hub_download(REPO_ID, name, local_dir=ASSETS_DIR)
                    got += 1
                except Exception:
                    pass
            if got == 0:
                sys.exit("Could not download encoder assets (config/tokenizer).")
            print(f"assets: downloaded {got} files -> {ASSETS_DIR}")
        convert_from_fp8(a.from_fp8, os.path.join(te_dir, OUT_NAME))
    else:
        weights_src = fetch_assets(a.src)
        stream_merge(weights_src, os.path.join(te_dir, OUT_NAME))
    print("\nAll set: encoder appears in the Rebels SeFi Loader dropdown after a ComfyUI restart.")
    print("Tokenizer/config assets are bundled in the node pack (encoder_assets/) - commit them to your repo.")


if __name__ == "__main__":
    main()
