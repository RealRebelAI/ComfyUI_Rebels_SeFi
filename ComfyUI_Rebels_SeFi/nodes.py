# ComfyUI_Rebels_SeFi - SeFi-Image (Semantic-First Diffusion) in ComfyUI
# Dropdown-driven: no path strings, no sefi_config.yaml required.
#   - transformer: merged .safetensors in models/diffusion_models  (dropdown)
#   - text encoder: Qwen3-VL HF folder under models/text_encoders  (dropdown)
#   - VAE: diffusers-format VAE folder under models/vae            (dropdown)
# Scale + semantic/texture channel split are DERIVED from the checkpoint and
# VAE config at load time. Optional sidecars (config.json / sefi_config.yaml
# next to the model file) are honored when present.
#
# Wraps the official SeFi-Image inference code (MIT):
#   https://github.com/jmliu206/SeFi-Image     (vendored in ./sefi_core)

import gc
import json
import os
import re
import types

import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

_SEFI_IMPORT_ERROR = None
try:
    from .sefi_core import SEFIRunnerDirect
    from .sefi_core.modeling import Flux2SEFITransformer2DModel, TextureLatentCodec
    from .sefi_core.modeling import qwen3vl_text_encoder as _qwen_mod
    from .encoder_loader import SeFiQwenEncoderFromFile, LazySeFiEncoder
    from .gguf_loader import iter_gguf_tensors, gguf_shapes, gguf_arch
except Exception as exc:  # noqa: BLE001
    _SEFI_IMPORT_ERROR = exc

# ---------------------------------------------------------------- fp8 support
_FP8_KEEP_BF16 = ("x_embedder", "context_embedder", "proj_out", "norm_out", "dual_time_embed")


def _fp8_linear_forward(self, x):
    weight = self.weight.to(x.dtype)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, weight, bias)


def _fp8_dtypes():
    return tuple(t for t in (getattr(torch, "float8_e4m3fn", None),
                             getattr(torch, "float8_e5m2", None)) if t is not None)


def _wrap_fp8_linears(transformer) -> int:
    """Attach dequant-at-forward wrappers to every Linear whose weight is fp8."""
    wrapped = 0
    fp8s = _fp8_dtypes()
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.dtype in fp8s:
            module.forward = types.MethodType(_fp8_linear_forward, module)
            wrapped += 1
    return wrapped


def _load_dtype_policy(weight_dtype: str):
    """Per-tensor storage dtype at load time. fp8 for big 2D weights, bf16 else."""
    use_fp8 = weight_dtype == "fp8_e4m3fn" and hasattr(torch, "float8_e4m3fn")

    def policy(key: str, tensor: torch.Tensor) -> torch.Tensor:
        if (use_fp8 and tensor.ndim == 2 and key.endswith(".weight")
                and not any(k in key for k in _FP8_KEEP_BF16)):
            return tensor.to(torch.float8_e4m3fn)
        return tensor.to(torch.bfloat16)
    return policy


def _assign_tensor(root: torch.nn.Module, dotted: str, tensor: torch.Tensor) -> bool:
    parts = dotted.split(".")
    mod = root
    for p in parts[:-1]:
        if p.isdigit():
            mod = mod[int(p)]
        elif hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return False
    leaf = parts[-1]
    if leaf in getattr(mod, "_parameters", {}):
        mod._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
        return True
    if leaf in getattr(mod, "_buffers", {}):
        mod._buffers[leaf] = tensor
        return True
    return False


def _stream_load_transformer(transformer, tensor_iter, needs_prefix: bool, weight_dtype: str):
    """Assign tensors one at a time into a meta-built transformer.
    Peak RAM ~= final model size (no giant state-dict double buffer)."""
    policy = _load_dtype_policy(weight_dtype)
    unexpected = []
    for key, tensor in tensor_iter:
        if needs_prefix and not key.startswith(("backbone.", "dual_time_embed.")):
            key = "backbone." + key
        if not _assign_tensor(transformer, key, policy(key, tensor)):
            unexpected.append(key)
    # anything still on meta was never loaded
    meta_left = [n for n, p in transformer.named_parameters() if p.is_meta]
    real_missing = [m for m in meta_left if "time_guidance_embed" not in m]
    if real_missing:
        raise ValueError(f"Checkpoint missing transformer keys (first 10): {real_missing[:10]}")
    for n, p in list(transformer.named_parameters()):
        if p.is_meta:  # tolerated leftovers -> zeros
            _assign_tensor(transformer, n, torch.zeros(p.shape, dtype=torch.bfloat16))
    for n, b in list(transformer.named_buffers()):
        if b.is_meta:
            _assign_tensor(transformer, n, torch.zeros(b.shape, dtype=torch.bfloat16))
            print(f"[Rebels SeFi] buffer {n} not in checkpoint - zero-initialized")
    if unexpected:
        print(f"[Rebels SeFi] ignoring unexpected keys (first 5): {unexpected[:5]}")


# ------------------------------------------------------------ block swap
def _collect_blocks(transformer):
    blocks = []
    backbone = transformer.backbone
    for name in ("transformer_blocks", "single_transformer_blocks"):
        ml = getattr(backbone, name, None)
        if ml is not None:
            blocks.extend(list(ml))
    return blocks


def _param_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _plan_residency(transformer, requested: int) -> int:
    """requested >=0 -> honor it. -1 -> auto from free VRAM."""
    blocks = _collect_blocks(transformer)
    if requested >= 0:
        return min(requested, len(blocks))
    if not torch.cuda.is_available():
        return len(blocks)
    free, _total = torch.cuda.mem_get_info()
    reserve = int(3.2 * 1024**3)          # activations @1024^2 (CFG x2) + decode + display
    small = _param_bytes(transformer) - sum(_param_bytes(b) for b in blocks)
    budget = free - reserve - small
    k, acc = 0, 0
    for b in blocks:
        acc += _param_bytes(b)
        if acc > budget:
            break
        k += 1
    print(f"[Rebels SeFi] auto residency: free={free/1e9:.1f}GB, "
          f"non-block={small/1e9:.2f}GB, resident {k}/{len(blocks)} blocks")
    return k


def _install_block_swap(transformer, device: str, blocks_on_gpu: int) -> int:
    """Stream transformer blocks GPU<->CPU per forward. Everything except the
    two block lists lives on GPU; the first `blocks_on_gpu` of each list stay
    resident, the rest ride hooks. Returns number of swapped blocks."""
    blocks = _collect_blocks(transformer)
    if blocks_on_gpu >= len(blocks):
        transformer.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0  # everything resident - no hooks, full speed

    # whole model to GPU first (small parts + resident blocks stay)
    transformer.to(device)

    def make_pre(block):
        def pre(module, args, kwargs=None):
            module.to(device)
            return None
        return pre

    def make_post(block):
        def post(module, args, output):
            module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        return post

    swapped = 0
    for i, block in enumerate(blocks):
        if i < blocks_on_gpu:
            continue
        block.to("cpu")
        block.register_forward_pre_hook(make_pre(block))
        block.register_forward_hook(make_post(block))
        swapped += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return swapped


# ------------------------------------------------------------- dropdown scans
def _scan_hf_folders(kind: str, must_contain: str) -> list[str]:
    names = []
    for root in folder_paths.get_folder_paths(kind):
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if not os.path.isdir(full):
                continue
            if os.path.isfile(os.path.join(full, must_contain)) or os.path.isfile(
                os.path.join(full, "vae", must_contain)
            ):
                names.append(entry)
    return names or ["(none found - see README)"]


def _resolve_hf_folder(kind: str, name: str) -> str:
    for root in folder_paths.get_folder_paths(kind):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(
        f"Folder '{name}' not found under models/{kind}. "
        "It must be an unpacked HuggingFace folder (config.json inside), not a single file."
    )


# ------------------------------------------------- checkpoint-derived config
_SCALE_PRESETS = {
    "0p5b": dict(attention_head_dim=128, num_attention_heads=12, num_layers=3, num_single_layers=10, joint_attention_dim=6144),
    "1b":   dict(attention_head_dim=128, num_attention_heads=16, num_layers=4, num_single_layers=12, joint_attention_dim=6144),
    "2b":   dict(attention_head_dim=128, num_attention_heads=20, num_layers=4, num_single_layers=16, joint_attention_dim=6144),
    "3b":   dict(attention_head_dim=128, num_attention_heads=22, num_layers=5, num_single_layers=18, joint_attention_dim=7680),
    "4b":   dict(attention_head_dim=128, num_attention_heads=24, num_layers=5, num_single_layers=20, joint_attention_dim=7680),
    "5b":   dict(attention_head_dim=128, num_attention_heads=26, num_layers=6, num_single_layers=21, joint_attention_dim=7680),
    "6b":   dict(attention_head_dim=128, num_attention_heads=28, num_layers=6, num_single_layers=22, joint_attention_dim=7680),
    "8b":   dict(attention_head_dim=128, num_attention_heads=30, num_layers=7, num_single_layers=24, joint_attention_dim=7680),
    "9b":   dict(attention_head_dim=128, num_attention_heads=32, num_layers=8, num_single_layers=24, joint_attention_dim=12288),
}
_QWEN_HIDDEN = {2048: "qwen3vl_2b", 2560: "qwen3vl_4b", 4096: "qwen3vl_8b"}


# ------------------------------------------------------------ single-file VAE
_PACK_DIR = os.path.dirname(os.path.abspath(__file__))


def _vae_is_flux2(keys) -> bool:
    return any(k.startswith("bn.") for k in keys)


def _derive_vae_config(shapes: dict) -> dict:
    """Reconstruct AutoencoderKL(-Flux2) architecture from weight shapes."""
    down_ids = [int(m.group(1)) for k in shapes
                if (m := re.match(r"encoder\.down_blocks\.(\d+)\.", k))]
    if not down_ids:
        raise ValueError(
            "This VAE file is not in diffusers format (no encoder.down_blocks keys) - "
            "SeFi needs the diffusers-format FLUX.2 VAE (e.g. flux2-vae.safetensors), "
            "not a ComfyUI first-stage VAE like ae.safetensors."
        )
    n_down = 1 + max(down_ids)
    block_out = []
    for i in range(n_down):
        outs = [shapes[k][0] for k in shapes
                if re.match(rf"encoder\.down_blocks\.{i}\.resnets\.\d+\.conv2\.weight$", k)]
        block_out.append(max(outs))
    layers_per_block = 1 + max(int(m.group(1)) for k in shapes
                               if (m := re.match(r"encoder\.down_blocks\.0\.resnets\.(\d+)\.", k)))
    in_channels = shapes["encoder.conv_in.weight"][1]
    latent_channels = shapes["encoder.conv_out.weight"][0] // 2
    return {
        "in_channels": int(in_channels),
        "out_channels": int(in_channels),
        "latent_channels": int(latent_channels),
        "block_out_channels": [int(x) for x in block_out],
        "layers_per_block": int(layers_per_block),
        "down_block_types": ["DownEncoderBlock2D"] * n_down,
        "up_block_types": ["UpDecoderBlock2D"] * n_down,
    }


def _resolve_vae_config(vae_path: str, keys, shapes: dict) -> tuple[dict, bool]:
    """Return (config_dict, is_flux2). Checks sidecar -> pack assets -> derive."""
    is_flux2 = _vae_is_flux2(keys)
    stem = os.path.splitext(os.path.basename(vae_path))[0]
    candidates = [
        os.path.splitext(vae_path)[0] + ".json",
        os.path.join(_PACK_DIR, "vae_assets", stem + ".json"),
        os.path.join(_PACK_DIR, "vae_assets", "flux2.json" if is_flux2 else "flux1.json"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                cfg.pop("_class_name", None)
                cfg.pop("_diffusers_version", None)
                print(f"[Rebels SeFi] VAE config: {cand}")
                return cfg, is_flux2
            except Exception:
                pass
    cfg = _derive_vae_config(shapes)
    print("[Rebels SeFi] VAE config derived from weights (drop the official "
          "config.json into vae_assets/ for guaranteed-exact values)")
    return cfg, is_flux2


def _peek_state_header(path: str) -> dict:
    from safetensors import safe_open

    shapes = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            shapes[k] = tuple(f.get_slice(k).get_shape())
    return shapes


def _block_count(shapes: dict, pattern: str) -> int:
    best = -1
    rx = re.compile(pattern)
    for k in shapes:
        m = rx.search(k)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def _derive_from_checkpoint(model_path: str, shapes: dict) -> dict:
    name_l = os.path.basename(model_path).lower()
    out = {"family": "turbo" if any(t in name_l for t in ("turbo", "distill", "dmd")) else "base"}

    n_double = _block_count(shapes, r"(?:^|\.)transformer_blocks\.(\d+)\.")
    n_single = _block_count(shapes, r"single_transformer_blocks\.(\d+)\.")
    scale = next(
        (s for s, p in _SCALE_PRESETS.items()
         if p["num_layers"] == n_double and p["num_single_layers"] == n_single),
        None,
    )
    if scale is None:
        m = re.search(r"([0-9]+(?:p[0-9]+)?b)", name_l)
        if m and m.group(1) in _SCALE_PRESETS:
            scale = m.group(1)
    if scale is None:
        raise ValueError(
            f"Cannot determine SeFi scale (double={n_double}, single={n_single}, file='{name_l}'). "
            "Rename the file to include its scale, e.g. 'SeFi-5B-...'."
        )
    out["scale"] = scale

    xk = next((k for k in shapes if k.endswith("x_embedder.weight")), None)
    if xk is None:
        raise ValueError("No x_embedder.weight in checkpoint - not a SeFi/Flux2 transformer?")
    # Flux2 packs latents 2x2 BEFORE x_embedder, so in_features IS the packed
    # channel count (semantic + texture*4). No division.
    out["total_channels"] = int(shapes[xk][1])
    out["has_wrapper_prefix"] = any(k.startswith("backbone.") for k in shapes)

    # optional sidecars next to the model file
    base_dir = os.path.dirname(model_path)
    stem = os.path.splitext(model_path)[0]
    out["transformer_config"] = None
    for cand in (stem + ".json", os.path.join(base_dir, "config.json")):
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                if isinstance(cfg, dict) and "Flux2" in str(cfg.get("_class_name", "Flux2")):
                    out["transformer_config"] = cfg
                    break
            except Exception:
                pass

    out["delta_t_sidecar"] = None
    out["shift_alpha_sidecar"] = None
    out["semantic_channels_sidecar"] = None
    _pack_cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sefi_configs")
    for cand in (
        os.path.join(base_dir, "sefi_config.yaml"),
        stem + ".yaml",
        os.path.join(_pack_cfg_dir, f"{out['scale']}-{out['family']}.yaml"),
    ):
        if os.path.isfile(cand):
            try:
                from omegaconf import OmegaConf

                y = OmegaConf.load(cand)
                container = OmegaConf.to_container(y, resolve=True)

                def _find(d, keys, path=""):
                    hits = []
                    if isinstance(d, dict):
                        for k, v in d.items():
                            p = f"{path}.{k}" if path else str(k)
                            if k in keys and isinstance(v, (int, float)):
                                hits.append((p, float(v)))
                            hits.extend(_find(v, keys, p))
                    return hits

                # official order: inference.delta_t, else training.sefi.delta_t_max,
                # else ANY delta_t / delta_t_max key anywhere in the file
                dt = OmegaConf.select(y, "inference.delta_t", default=None)
                src = "inference.delta_t"
                if dt is None:
                    dt = OmegaConf.select(y, "training.sefi.delta_t_max", default=None)
                    src = "training.sefi.delta_t_max"
                if dt is None:
                    found = _find(container, {"delta_t"}) or _find(container, {"delta_t_max"})
                    if found:
                        src, dt = found[0]
                if dt is not None:
                    out["delta_t_sidecar"] = float(dt)
                    print(f"[Rebels SeFi] delta_t {float(dt):.4f} from yaml key '{src}'")
                else:
                    print(f"[Rebels SeFi] WARNING: {cand} has no delta_t/delta_t_max key - "
                          "delta_t will use the node widget value")
                al = OmegaConf.select(y, "inference.timestep_shift_alpha", default=None)
                if al is None:
                    found = _find(container, {"timestep_shift_alpha"})
                    if found:
                        al = found[0][1]
                out["shift_alpha_sidecar"] = float(al) if al is not None else None
                sem = OmegaConf.select(y, "model.semantic_channels", default=None)
                if sem is not None:
                    out["semantic_channels_sidecar"] = int(sem)
                if out["delta_t_sidecar"] is not None or out["shift_alpha_sidecar"] is not None:
                    print(f"[Rebels SeFi] using SeFi config: {cand}")
                    break
            except Exception:
                pass
    return out


def _build_transformer_cfg(derived: dict, total_channels: int, text_output_dim: int) -> dict:
    cfg = dict(derived["transformer_config"] or {})
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.update(_SCALE_PRESETS[derived["scale"]])
    cfg["in_channels"] = int(total_channels)
    cfg["out_channels"] = int(total_channels)
    cfg["guidance_embeds"] = False
    if int(cfg["joint_attention_dim"]) != int(text_output_dim):
        raise ValueError(
            f"Text encoder output dim {text_output_dim} != transformer joint_attention_dim "
            f"{cfg['joint_attention_dim']}. Wrong Qwen3-VL size for this checkpoint."
        )
    return cfg


_PIPE_CACHE = {}


class RebelsSeFiLoader:
    @classmethod
    def _model_choices(cls):
        names = []
        for kind in ("diffusion_models", "unet"):
            try:
                for f in folder_paths.get_filename_list(kind):
                    if f not in names:
                        names.append(f)
            except Exception:
                pass
            try:
                roots = folder_paths.get_folder_paths(kind)
            except Exception:
                continue
            for root in roots:
                if os.path.isdir(root):
                    for f in sorted(os.listdir(root)):
                        if f.lower().endswith((".gguf", ".safetensors", ".sft")) and f not in names:
                            names.append(f)
        return sorted(names) or ["(none found)"]

    @classmethod
    def _vae_choices(cls):
        files = [f for f in folder_paths.get_filename_list("vae")
                 if f.lower().endswith((".safetensors", ".sft"))]
        folders = [f"[folder] {n}" for n in _scan_hf_folders("vae", "config.json")
                   if not n.startswith("(none")]
        return (files + folders) or ["(none found - see README)"]

    @classmethod
    def _te_choices(cls):
        files = [f for f in folder_paths.get_filename_list("text_encoders")
                 if f.lower().endswith((".safetensors", ".sft"))]
        folders = [f"[folder] {n}" for n in _scan_hf_folders("text_encoders", "config.json")
                   if not n.startswith("(none")]
        return (files + folders) or ["(none found - run prepare_sefi_encoder.py)"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (s._model_choices(),),
                "text_encoder": (s._te_choices(),),
                "vae": (s._vae_choices(),),
                "weight_dtype": (["fp8_e4m3fn", "bf16"], {"default": "fp8_e4m3fn"}),
                "text_encoder_device": (["cpu", "cuda"], {"default": "cpu"}),
                "blocks_on_gpu": ("INT", {"default": -1, "min": -1, "max": 64,
                                          "tooltip": "-1 = AUTO: measures free VRAM and keeps as many blocks resident as safely fit (full-GPU if everything fits, streams the rest). Set a number to override."}),
                "unload_encoder_after_encode": ("BOOLEAN", {"default": True,
                                                            "tooltip": "Free the 8.8GB Qwen3-VL from RAM right after encoding (needed on 16GB-RAM machines). Embeddings are cached per prompt."}),
                "delta_t": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.005,
                                      "tooltip": "-1 = auto (yaml sidecar, else 0.1 fallback). Set a value to OVERRIDE the yaml - explicit always wins."}),
                "timestep_shift_alpha": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 10.0, "step": 0.01,
                                                   "tooltip": "-1 = auto (yaml sidecar, else 0.3 Base / 1.0 Turbo)."}),
            }
        }

    RETURN_TYPES = ("SEFI_PIPE",)
    RETURN_NAMES = ("sefi_pipe",)
    FUNCTION = "load"
    CATEGORY = "Rebels/SeFi"
    TITLE = "Rebels SeFi Loader"

    def load(self, model_name, text_encoder, vae, weight_dtype, text_encoder_device, blocks_on_gpu, unload_encoder_after_encode, delta_t, timestep_shift_alpha):
        if _SEFI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "SeFi core failed to import - diffusers/transformers too old for FLUX.2 / Qwen3-VL classes. Run:\n"
                "  python_embeded\\python.exe -m pip install -U diffusers transformers omegaconf accelerate\n"
                f"Original error: {_SEFI_IMPORT_ERROR}"
            )

        key = (model_name, text_encoder, vae, weight_dtype, text_encoder_device,
               int(blocks_on_gpu), bool(unload_encoder_after_encode),
               float(delta_t), float(timestep_shift_alpha))
        if key in _PIPE_CACHE:
            return (_PIPE_CACHE[key],)
        if _PIPE_CACHE:
            _PIPE_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        from safetensors.torch import load_file

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = None
        for kind in ("diffusion_models", "unet"):
            try:
                model_path = folder_paths.get_full_path(kind, model_name)
            except Exception:
                model_path = None
            if model_path:
                break
            try:
                for root in folder_paths.get_folder_paths(kind):
                    cand = os.path.join(root, model_name)
                    if os.path.isfile(cand):
                        model_path = cand
                        break
            except Exception:
                pass
            if model_path:
                break
        if not model_path:
            raise FileNotFoundError(f"Could not resolve {model_name} in diffusion_models/unet folders.")
        is_gguf = model_path.lower().endswith(".gguf")
        if is_gguf:
            arch = gguf_arch(model_path)
            if arch and arch != "sefi":
                print(f"[Rebels SeFi] warning: GGUF arch tag is '{arch}' (expected 'sefi')")
            shapes = gguf_shapes(model_path)
        else:
            shapes = _peek_state_header(model_path)
        derived = _derive_from_checkpoint(model_path, shapes)

        # ---- VAE (single file from dropdown; HF folders as "[folder] ...") ----
        if vae.startswith("[folder] "):
            vae_root = _resolve_hf_folder("vae", vae[len("[folder] "):])
            sub = "vae" if os.path.isfile(os.path.join(vae_root, "vae", "config.json")) else None
            with open(os.path.join(vae_root, sub or "", "config.json"), "r", encoding="utf-8") as fh:
                vae_cfg = json.load(fh)
            is_flux2 = "Flux2" in str(vae_cfg.get("_class_name", ""))
            if is_flux2:
                from diffusers.models import AutoencoderKLFlux2 as VAECls
            else:
                from diffusers.models import AutoencoderKL as VAECls
            kw = {"torch_dtype": torch.bfloat16, "local_files_only": True}
            texture_vae = (VAECls.from_pretrained(vae_root, subfolder=sub, **kw) if sub
                           else VAECls.from_pretrained(vae_root, **kw))
        else:
            vae_path = folder_paths.get_full_path("vae", vae)
            vae_shapes = _peek_state_header(vae_path)
            vae_cfg, is_flux2 = _resolve_vae_config(vae_path, vae_shapes.keys(), vae_shapes)
            if is_flux2:
                from diffusers.models import AutoencoderKLFlux2 as VAECls
            else:
                from diffusers.models import AutoencoderKL as VAECls
            texture_vae = VAECls.from_config(vae_cfg)
            vstate = load_file(vae_path)
            vmissing, vunexpected = texture_vae.load_state_dict(vstate, strict=False)
            del vstate
            if vmissing:
                raise ValueError(
                    f"VAE file missing keys for {VAECls.__name__} (first 10): {vmissing[:10]}. "
                    "Wrong VAE selected, or add the official config.json to vae_assets/."
                )
            if vunexpected:
                print(f"[Rebels SeFi] VAE: ignoring unexpected keys (first 5): {vunexpected[:5]}")
            texture_vae = texture_vae.to(torch.bfloat16)
        texture_codec = TextureLatentCodec(texture_vae=texture_vae,
                                           texture_vae_name="flux2" if is_flux2 else "flux1")
        texture_channels = int(texture_codec.texture_channels)
        semantic_channels = derived["semantic_channels_sidecar"] or (derived["total_channels"] - texture_channels)
        if semantic_channels <= 0:
            raise ValueError(
                f"Channel derivation failed (total={derived['total_channels']}, texture={texture_channels}). "
                "Wrong VAE for this checkpoint?"
            )

        # ---- Qwen3-VL text encoder (file dropdown; HF folder supported as "[folder] ...") ----
        preset = _SCALE_PRESETS[derived["scale"]]
        te_model_name = _QWEN_HIDDEN.get(preset["joint_attention_dim"] // 3)
        if te_model_name is None:
            raise ValueError(f"Unexpected joint_attention_dim for scale {derived['scale']}.")
        if text_encoder.startswith("[folder] "):
            te_root = _resolve_hf_folder("text_encoders", text_encoder[len("[folder] "):])
            te_parent, te_dirname = os.path.split(te_root.rstrip("/\\"))
            _qwen_mod.QWEN3VL_MODEL_PATHS[te_model_name] = te_dirname

            def _build_te():
                return _qwen_mod.Qwen3VLTextEncoder(
                    model_name=te_model_name, weights_root=te_parent,
                    max_length=512, hidden_layers=(9, 18, 27), torch_dtype=torch.bfloat16,
                ).to("cpu").eval()
        else:
            te_path = folder_paths.get_full_path("text_encoders", text_encoder)

            def _build_te():
                return SeFiQwenEncoderFromFile(
                    weights_path=te_path, model_name=te_model_name,
                    max_length=512, hidden_layers=(9, 18, 27), torch_dtype=torch.bfloat16,
                ).to("cpu").eval()
        te = LazySeFiEncoder(_build_te, unload_after_encode=False)  # freed at step 1 by the sampler
        te.output_dim = int(preset["joint_attention_dim"])  # known a priori - no probe load

        # ---- transformer ----
        tcfg = _build_transformer_cfg(derived, derived["total_channels"], int(te.output_dim))
        with torch.device("meta"):
            transformer = Flux2SEFITransformer2DModel(backbone_config=tcfg, text_input_dim=int(te.output_dim))

        if is_gguf:
            src_iter = ((name, ten) for name, ten, _ in iter_gguf_tensors(model_path))
        else:
            from safetensors import safe_open

            def _st_iter():
                with safe_open(model_path, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        yield k, f.get_tensor(k)
            src_iter = _st_iter()

        print("[Rebels SeFi] streaming transformer weights (one tensor at a time)...")
        _stream_load_transformer(transformer, src_iter,
                                 needs_prefix=not derived["has_wrapper_prefix"],
                                 weight_dtype=weight_dtype)
        transformer = transformer.eval()
        n_fp8 = _wrap_fp8_linears(transformer)
        if n_fp8:
            print(f"[Rebels SeFi] {n_fp8} Linear layers in fp8_e4m3fn storage (bf16 compute)")
        gc.collect()

        if device == "cuda":
            resident = _plan_residency(transformer, int(blocks_on_gpu))
            swapped = _install_block_swap(transformer, device, resident)
            if swapped:
                print(f"[Rebels SeFi] block swap: {swapped} blocks streaming CPU<->GPU, {resident} resident")
            else:
                print("[Rebels SeFi] entire transformer resident on GPU (no offloading needed)")
        else:
            transformer = transformer.to(device)

        texture_codec = texture_codec.to(device=device).eval()

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

        if delta_t >= 0:
            resolved_delta = float(delta_t)
            print(f"[Rebels SeFi] delta_t={resolved_delta:.4f} (USER widget override)")
        elif derived["delta_t_sidecar"] is not None:
            resolved_delta = float(derived["delta_t_sidecar"])
            print(f"[Rebels SeFi] delta_t={resolved_delta:.4f} (yaml)")
        else:
            resolved_delta = 0.1
            print("[Rebels SeFi] delta_t=0.1000 (FALLBACK - no yaml value found; consider setting it)")
        if timestep_shift_alpha >= 0:
            resolved_alpha = float(timestep_shift_alpha)
        elif derived["shift_alpha_sidecar"] is not None:
            resolved_alpha = float(derived["shift_alpha_sidecar"])
        else:
            resolved_alpha = 1.0 if derived["family"] == "turbo" else 0.3

        runner = SEFIRunnerDirect(
            transformer=transformer,
            text_encoder=te,
            texture_codec=texture_codec,
            noise_scheduler=scheduler,
            pipeline_cls=Flux2KleinPipeline,
            semantic_channels=semantic_channels,
            texture_channels=texture_channels,
            device=device,
            weight_dtype=torch.bfloat16,
            delta_t=resolved_delta,
            timestep_shift_alpha=resolved_alpha,
        )

        pipe = {"runner": runner, "family": derived["family"], "scale": derived["scale"],
                "unload_encoder": bool(unload_encoder_after_encode)}
        _PIPE_CACHE[key] = pipe
        print(
            f"[Rebels SeFi] loaded {model_name} | scale={derived['scale']} family={derived['family']} "
            f"| sem/tex channels={semantic_channels}/{texture_channels} "
            f"| delta_t={resolved_delta:.4f} alpha={resolved_alpha:.2f}"
        )
        return (pipe,)


class RebelsSeFiSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sefi_pipe": ("SEFI_PIPE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200,
                                  "tooltip": "0 = default (50 Base, 4 Turbo). Turbo allows 4/8/10 only."}),
                "guidance_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0, "step": 0.1,
                                             "tooltip": "-1 = default (4.0 Base, 1.0 Turbo)."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Rebels/SeFi"
    TITLE = "Rebels SeFi Sampler"

    def sample(self, sefi_pipe, prompt, steps, guidance_scale, width, height, seed):
        runner = sefi_pipe["runner"]
        turbo = sefi_pipe["family"] == "turbo"

        resolved_steps = int(steps) if steps > 0 else (4 if turbo else 50)
        resolved_guidance = float(guidance_scale) if guidance_scale >= 0 else (1.0 if turbo else 4.0)
        if turbo:
            if resolved_steps not in (4, 8, 10):
                raise ValueError(f"SeFi Turbo supports 4/8/10 steps, got {resolved_steps} (0 = default 4).")
            if resolved_guidance != 1.0:
                print(f"[Rebels SeFi] WARNING: Turbo is distilled for guidance 1.0; running your "
                      f"requested {resolved_guidance} anyway (2x slower, may degrade quality)")

        try:
            import comfy.model_management as mm
        except Exception:
            mm = None
        if mm is not None:
            try:
                mm.unload_all_models()
                mm.soft_empty_cache()
            except Exception:
                pass

        import time as _time
        _t = {"last": None}
        try:
            from comfy.utils import ProgressBar
            _pbar = ProgressBar(resolved_steps)
        except Exception:
            _pbar = None

        def _on_step(i, total):
            if i == 0 and sefi_pipe.get("unload_encoder") and hasattr(runner.text_encoder, "free"):
                runner.text_encoder.free()  # both prompts encoded by now; drop 8.8GB before the loop
            now = _time.time()
            per = f" ({now - _t['last']:.1f}s/step)" if _t["last"] is not None else ""
            _t["last"] = now
            print(f"[Rebels SeFi] step {i + 1}/{total}{per}", flush=True)
            if _pbar is not None:
                _pbar.update_absolute(i + 1, total)

        runner.step_callback = _on_step
        runner.transformer.eval()
        runner.texture_codec.eval()

        generator = torch.Generator(device=str(runner.device)).manual_seed(int(seed))
        try:
            images = runner.generate_batch(
                prompts=[prompt],
                num_inference_steps=resolved_steps,
                guidance_scale=resolved_guidance,
                height=int(height),
                width=int(width),
                generator=generator,
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            if mm is not None:
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
        out = [torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0) for img in images]
        return (torch.stack(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "RebelsSeFiLoader": RebelsSeFiLoader,
    "RebelsSeFiSampler": RebelsSeFiSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsSeFiLoader": "Rebels SeFi Loader",
    "RebelsSeFiSampler": "Rebels SeFi Sampler",
}
