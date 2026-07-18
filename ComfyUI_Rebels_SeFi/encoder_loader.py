# encoder_loader.py - build the SeFi Qwen3-VL text encoder from:
#   - a SINGLE merged .safetensors picked from the models/text_encoders dropdown
#   - tokenizer/config assets bundled inside this node pack (encoder_assets/)
# Mirrors the vendored Qwen3VLTextEncoder.encode() exactly (chat template,
# max_length padding, hidden layers 9/18/27 concatenated).

import os
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .device_compat import empty_cache

PACK_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = {
    "qwen3vl_2b": os.path.join(PACK_DIR, "encoder_assets", "qwen3vl_2b"),
    "qwen3vl_4b": os.path.join(PACK_DIR, "encoder_assets", "qwen3vl_4b"),
    "qwen3vl_8b": os.path.join(PACK_DIR, "encoder_assets", "qwen3vl_8b"),
}


class SeFiQwenEncoderFromFile(nn.Module):
    """Qwen3-VL text encoder from one merged weights file + bundled assets."""

    def __init__(
        self,
        weights_path: str,
        model_name: str = "qwen3vl_4b",
        max_length: int = 512,
        hidden_layers: Sequence[int] = (9, 18, 27),
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoTokenizer, Qwen3VLForConditionalGeneration

        assets_dir = ASSETS.get(model_name)
        if assets_dir is None:
            raise ValueError(f"Unknown encoder model_name {model_name}.")
        if not os.path.isfile(os.path.join(assets_dir, "config.json")):
            # Self-heal: pack updates can wipe generated assets. The weights file
            # is untouched (lives in models/text_encoders), so just re-fetch the
            # tiny config/tokenizer files.
            _REPOS = {
                "qwen3vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
                "qwen3vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
                "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
            }
            _FILES = ["config.json", "generation_config.json", "tokenizer_config.json",
                      "tokenizer.json", "vocab.json", "merges.txt",
                      "preprocessor_config.json", "chat_template.json",
                      "special_tokens_map.json"]
            print(f"[Rebels SeFi] encoder assets missing for {model_name} - fetching (~few MB, one time)")
            try:
                from huggingface_hub import hf_hub_download
                os.makedirs(assets_dir, exist_ok=True)
                got = 0
                for name in _FILES:
                    try:
                        hf_hub_download(_REPOS[model_name], name, local_dir=assets_dir)
                        got += 1
                    except Exception:
                        pass  # not every repo ships every file
                print(f"[Rebels SeFi] fetched {got} asset files -> {assets_dir}")
            except Exception as exc:  # noqa: BLE001
                print(f"[Rebels SeFi] asset auto-fetch failed: {exc}")
        if not os.path.isfile(os.path.join(assets_dir, "config.json")):
            raise FileNotFoundError(
                f"Encoder assets missing for {model_name} (expected {assets_dir}/config.json) "
                "and auto-download failed (offline?). Run prepare_sefi_encoder.py once, or copy "
                "the Qwen3-VL config/tokenizer files into that folder manually."
            )

        self.model_name = model_name
        self.max_length = int(max_length)
        self.hidden_layers = tuple(int(x) for x in hidden_layers)

        self.tokenizer = AutoTokenizer.from_pretrained(assets_dir, local_files_only=True)
        cfg = AutoConfig.from_pretrained(assets_dir, local_files_only=True)

        # NORMAL init on CPU (NOT meta/to_empty): non-persistent buffers such as
        # rotary inv_freq are computed in __init__ and are absent from weight
        # files. meta+to_empty materializes them as uninitialized memory - fine
        # on the FIRST build of a process (fresh zeroed pages), garbage on any
        # REBUILD (recycled heap) -> corrupted positional encoding -> hot
        # embeddings -> noise images on every prompt change. Init properly,
        # then stream weights in one tensor at a time to keep RAM at ~1x model.
        model = Qwen3VLForConditionalGeneration._from_config(cfg, torch_dtype=torch_dtype)
        model = model.to("cpu")

        from safetensors import safe_open
        remaining = {n for n, _ in model.named_parameters()}
        unexpected = []
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                t = f.get_tensor(key).to(torch_dtype)
                parts = key.split(".")
                mod, ok = model, True
                for p in parts[:-1]:
                    if p.isdigit():
                        mod = mod[int(p)]
                    elif hasattr(mod, p):
                        mod = getattr(mod, p)
                    else:
                        ok = False
                        break
                leaf = parts[-1]
                if ok and leaf in getattr(mod, "_parameters", {}):
                    mod._parameters[leaf] = torch.nn.Parameter(t, requires_grad=False)
                    remaining.discard(key)
                elif ok and leaf in getattr(mod, "_buffers", {}):
                    mod._buffers[leaf] = t
                    remaining.discard(key)
                else:
                    unexpected.append(key)
            # tied head: absent from fp8/stripped merges, never used for encoding,
            # but tie it to embeddings so it's real memory either way
            if "lm_head.weight" in remaining and "model.embed_tokens.weight" not in remaining:
                model.lm_head.weight = model.get_input_embeddings().weight
                remaining.discard("lm_head.weight")
                print("[Rebels SeFi] Info: tied missing lm_head.weight to embed_tokens.")
        missing = sorted(remaining)
        
        # Visual tower was stripped at merge time - those missing keys are expected.
        # We also explicitly ignore lm_head.weight here just in case the wiring logic 
        # above couldn't find embed_tokens.
        real_missing = [
            m for m in missing 
            if ".visual." not in m and not m.startswith("visual.") and m != "lm_head.weight"
        ]
        
        if real_missing:
            raise ValueError(
                f"Merged encoder file is missing text-model keys (first 10): {real_missing[:10]}. "
                "Re-run prepare_sefi_encoder.py - the merge may be incomplete."
            )
        if unexpected:
            print(f"[Rebels SeFi] encoder: ignoring unexpected keys (first 5): {unexpected[:5]}")

        if hasattr(model, "model") and hasattr(model.model, "visual"):
            del model.model.visual
        self.model = model.eval()

        self.output_dim = int(self.model.config.text_config.hidden_size) * len(self.hidden_layers)

    # ---- identical behavior to the vendored encoder ----
    def _build_chat_text(self, caption: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": caption}]}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    @staticmethod
    def _prepare_text_ids(x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        device = x.device
        out_ids = []
        for _ in range(batch):
            # Explicitly cast to the target device immediately to prevent CPU/GPU mixing
            t = torch.arange(1, device=device)
            h = torch.arange(1, device=device)
            w = torch.arange(1, device=device)
            l = torch.arange(seq_len, device=device)
            out_ids.append(torch.cartesian_prod(t, h, w, l))
        return torch.stack(out_ids)

    @torch.no_grad()
    def encode(self, captions: list[str], dtype: torch.dtype | None = None):
        device = next(self.model.parameters()).device
        if dtype is None:
            dtype = next(self.model.parameters()).dtype

        # --- DEVICE MISMATCH FIX ---
        # ComfyUI shuffles parameters to VRAM, but often leaves uninitialized 
        # buffers (like Qwen's rotary embeddings) stranded on the CPU.
        # This sweeps the entire module to ensure no buffer is left behind.
        self.model.to(device)
        # ---------------------------

        chat_texts = [self._build_chat_text(c) for c in captions]
        tok = self.tokenizer(
            chat_texts, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length,
        )
        output = self.model.model(
            input_ids=tok["input_ids"].to(device),
            attention_mask=tok["attention_mask"].to(device),
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        hs = output.hidden_states
        max_idx = len(hs) - 1
        for li in self.hidden_layers:
            if li > max_idx:
                raise ValueError(f"Hidden layer {li} requested but model provides up to {max_idx}.")
        stacked = torch.stack([hs[i] for i in self.hidden_layers], dim=1).to(dtype=dtype)
        b, nl, sl, hd = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, sl, nl * hd)
        text_ids = self._prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids


class LazySeFiEncoder(torch.nn.Module):
    """Builds the Qwen3-VL encoder ON DEMAND, encodes, and (optionally) frees it.

    On 16GB-RAM machines the 8.8GB encoder cannot stay resident next to the
    CPU-offloaded transformer blocks, so: build -> encode -> free. Embeddings
    are cached per prompt so re-queues with the same prompt never reload it.
    """

    def __init__(self, builder, unload_after_encode: bool = True):
        super().__init__()
        self._builder = builder            # zero-arg callable -> encoder module
        self._unload = unload_after_encode
        self._enc = None
        self._cache = {}
        self.output_dim = None

    def _ensure(self):
        if self._enc is None:
            print("[Rebels SeFi] loading text encoder (on demand)...")
            self._enc = self._builder()
            self.output_dim = self._enc.output_dim
        return self._enc

    def probe_output_dim(self) -> int:
        if self.output_dim is None:
            self._ensure()
            if self._unload:
                self.free()
        return self.output_dim

    def free(self):
        if self._enc is not None:
            self._enc = None
            import gc
            gc.collect()
            empty_cache()
            print("[Rebels SeFi] text encoder freed from RAM")

    @torch.no_grad()
    def encode(self, captions, dtype=None):
        key = (tuple(captions), str(dtype))
        if key in self._cache:
            print("[Rebels SeFi] using cached prompt embeddings")
            v = self._cache[key]
            return (v[0].clone(), v[1].clone())
        enc = self._ensure()
        self.output_dim = enc.output_dim
        out = enc.encode(captions, dtype=dtype)
        emb = out[0]
        if not torch.isfinite(emb).all():
            raise RuntimeError(
                "[Rebels SeFi] text encoder produced NaN/Inf embeddings on this encode "
                "(in-session rebuild bug confirmed). Restart ComfyUI and report this."
            )
        print(f"[Rebels SeFi] embeds ok: shape={tuple(emb.shape)} "
              f"mean={emb.float().mean().item():.4f} std={emb.float().std().item():.4f}")
        # store CPU copies; hand out fresh clones every call. Cached tensors are
        # shared across generations - if anything downstream mutates them
        # in-place, outputs degrade run over run until restart. Clones make the
        # cache immutable by construction.
        val = (out[0].detach().to("cpu"), out[1].detach().to("cpu"))
        self._cache[key] = val
        while len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        if self._unload:
            self.free()
        return (val[0].clone(), val[1].clone())
