# ComfyUI Rebels SeFi (SeFi-Image)

Run **SeFi-Image 5B** (Semantic-First Diffusion, FLUX.2-Klein-based) in ComfyUI — **Base and Turbo, safetensors (bf16) and GGUF** — engineered to run on **8GB VRAM / 16GB RAM** with automatic VRAM-aware offloading.

Wraps the official MIT inference code from [jmliu206/SeFi-Image](https://github.com/jmliu206/SeFi-Image) (vendored in `sefi_core/`, credit SeFi-Team). Nodes, GGUF support, and memory management by [realrebelai](https://github.com/RealRebelAI).

## What SeFi is

One transformer, one latent — but the latent carries **semantic + texture channel groups on two staggered timesteps**: the semantic stream denoises `delta_t` ahead and anchors structure while the texture stream fills in detail. Only the texture channels are decoded through the VAE. Text encoder is **Qwen3-VL-4B**.

## Install

1. Drop this folder into `ComfyUI/custom_nodes/` and restart.
2. Update dependencies (the FLUX.2 classes need a recent diffusers):
```
python_embeded\python.exe -m pip install -U diffusers transformers omegaconf accelerate
```

## Get the models

Everything you need is in one place: [realrebelai/SeFi-Image-5B-Base](https://huggingface.co/realrebelai/SeFi-Image-5B-Base/tree/main)

[realrebelai/SeFi-Image-5B-Turbo](https://huggingface.co/realrebelai/SeFi-Image-5B-Turbo/tree/main)

| File | Goes in |
|------|---------|
| `SeFi-5B-Base_transformer_bf16.safetensors` / `SeFi-5B-Turbo_transformer_bf16.safetensors` (full quality) **or** any `SeFi-5B-*-Q4_0.gguf` … `Q8_0.gguf` (smaller download, same dropdown) | `models/diffusion_models/` (or `models/unet/`) |
| `SeFi_Qwen3-VL-4B_text_bf16.safetensors` (text encoder) | `models/text_encoders/` |
| `sefi_vae.safetensors` (VAE) | `models/vae/` |

Keep the scale + family in the transformer filename (`5B`, `Base`/`Turbo`) — the loader auto-detects both from it.

⚠️ **Use the SeFi VAE from the repo, not a generic FLUX.2 VAE.** A generic one will *load* but produces degraded output — SeFi's texture stream is trained against its own VAE's latent statistics. They are **not interchangeable**.

<details>
<summary>Prefer to build the text encoder yourself?</summary>

`prepare_sefi_encoder.py` builds the same single-file encoder locally — from the official Qwen repo:
```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_Rebels_SeFi\prepare_sefi_encoder.py --comfy ComfyUI
```
or converted from Comfy-Org's `qwen3vl_4b_fp8_scaled.safetensors` you may already have (skips the 9GB download):
```
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_Rebels_SeFi\prepare_sefi_encoder.py --comfy ComfyUI --from-fp8 ComfyUI\models\text_encoders\qwen3vl_4b_fp8_scaled.safetensors
```
</details>

## Nodes (Rebels → SeFi)

**Rebels SeFi Loader** — three dropdowns, all single files from standard ComfyUI folders: transformer (`diffusion_models`/`unet`, safetensors **or** GGUF), text encoder (`text_encoders`), VAE (`vae`). Scale, Base/Turbo family, and the semantic/texture channel split are read from the checkpoint automatically.

- `weight_dtype`: **bf16** (recommended, full quality) or `fp8_e4m3fn` (~half the memory, experimental).
- `blocks_on_gpu`: **-1 = AUTO** — measures your free VRAM at load and keeps as many transformer blocks resident as safely fit; the rest stream CPU↔GPU per step. Full-GPU speed automatically on big cards, works down to 8GB. Set a number to override.
- `text_encoder_device`: `cpu` (default). The encoder loads on demand, encodes, and is freed from RAM before sampling starts — embeddings are cached per prompt, so re-running the same prompt skips the reload entirely.
- `unload_encoder_after_encode`: keep **on** for 16GB-RAM machines.
- `delta_t` / `timestep_shift_alpha`: **-1 = auto** (reads the model's `sefi_config.yaml` from `sefi_configs/`, else sane defaults: alpha 0.3 Base / 1.0 Turbo). Setting an explicit value always overrides the yaml — the console prints which source won.

**Rebels SeFi Sampler** — prompt, steps (0 = default: **50 Base / 4 Turbo**), guidance (-1 = default: **4.0 Base / 1.0 Turbo**), size (multiples of 16), seed → IMAGE. Console shows live step progress with per-step timing.

Turbo is distilled for **4/8/10 steps at guidance 1.0**. Other guidance values are allowed but warned — expect slower runs and possible quality loss.

## 8GB VRAM / 16GB RAM notes

- Turbo runs ~**5s/step** on an RTX 3070 with cached embeddings (~21s per image after the first gen of a prompt).
- The first generation of each new prompt is slower: the Qwen3-VL encoder loads, encodes on CPU, and frees itself.
- Weights stream into the model one tensor at a time at load — no giant RAM spike.

## Bundled configs

`sefi_configs/` holds the models' `sefi_config.yaml` files (`5b-base.yaml`, `5b-turbo.yaml`) so `delta_t`/alpha resolve automatically. `encoder_assets/` holds the Qwen3-VL tokenizer/configs (auto-restored if missing). **If you update this pack by overwriting the folder, keep these directories** — or just `git pull`.

## Extras

- `prepare_sefi_encoder.py` — encoder builder (see above).
- `sefi_merge_transformer.py` — streams sharded diffusers transformers into one safetensors (<1GB RAM), for making your own single-file models.

## Troubleshooting

- **"(none found)" in a dropdown** → the file isn't in the matching models folder, or (VAE/encoder) it's not the expected format — see Get the models.
- **Noise / grey output on the second prompt of a session** → you're running an old `encoder_loader.py`; update the pack.
- **Console says `delta_t=... (FALLBACK)`** → the yaml for that model family is missing from `sefi_configs/`.

## License

Node pack + vendored inference code: **MIT**. Model weights: **CC BY-NC 4.0 (non-commercial)** — respect the SeFi-Image license.
