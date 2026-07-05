# ComfyUI Rebels SeFi (SeFi-Image)

Run **SeFi-Image** (Semantic-First Diffusion, FLUX.2-Klein-based) in ComfyUI — Base, RL, and Turbo checkpoints — tuned for **8GB VRAM / 16GB RAM**.

Wraps the official MIT inference code: https://github.com/jmliu206/SeFi-Image (vendored in `sefi_core/`, credit SeFi-Team). Node wrapper by realrebelai.

## What SeFi is
One transformer, one latent — but the latent carries **semantic + texture channel groups on two staggered timesteps**: semantic structure denoises `delta_t` ahead and anchors the texture stream. Only the texture channels are decoded (Flux VAE). Text encoder is **Qwen3-VL** (bundled in the checkpoint).

## Install
1. Drop this folder into `ComfyUI/custom_nodes/`.
2. Update deps (FLUX.2 classes need recent diffusers):
```
python_embeded\python.exe -m pip install -U diffusers transformers omegaconf accelerate
```
(diffusers >= 0.39 required)
3. Download a checkpoint repo (gated — accept the license on HF first), e.g. `SeFi-Image/SeFi-Image-5B-Base`, keeping the folder layout intact (`sefi_config.yaml`, `transformer/`, `vae/`, `text_encoder/`, `scheduler/`).

## Model installation (standard ComfyUI folders)
**Models**: https://huggingface.co/realrebelai/SeFi-Image-5B-Base/tree/main

 **VAE**: your existing single-file `flux2-vae.safetensors` in `models/vae/` works directly - it appears in the dropdown. For guaranteed-exact config values, drop the official `config.json` (from the SeFi repo's `vae/` folder) into this pack's `vae_assets/` as `flux2.json`; without it the loader derives the architecture from the weights. Diffusers-format VAE folders also work (listed as `[folder] ...`).

Optional but recommended: copy the repo's `sefi_config.yaml` and the transformer `config.json` next to the merged file - the loader reads them for exact `delta_t` / shift values. Without them it uses sensible derived defaults.

## Nodes (Rebels → SeFi)
**Rebels SeFi Loader** — three dropdowns: transformer (diffusion_models), Qwen3-VL folder (text_encoders), VAE folder (vae). Scale, Base/Turbo family, and the semantic/texture channel split are derived automatically from the checkpoint.
- `weight_dtype`: **fp8_e4m3fn** (default, ~5.2GB VRAM for 5B — use this on 8GB cards) or bf16 (~10.4GB, needs 12GB+). fp8 keeps embedders/proj_out/time-embeds in bf16 for quality.
- `text_encoder_device`: **cpu** (default; Qwen3-VL encodes once per prompt, no VRAM cost) or cuda.
- `delta_t` / `timestep_shift_alpha`: −1 = model defaults. Advanced dials — Base/RL default alpha 0.3, Turbo 1.0.

**Rebels SeFi Sampler** — prompt, steps (0 = model default: 50 Base/RL, **4 Turbo**), guidance (−1 = default: 4.0 Base/RL, **1.0 Turbo**), size (multiples of 16), seed → IMAGE.

Turbo checkpoints only support 4/8/10 steps and guidance 1.0 (enforced).

## 8GB VRAM notes
- 5B + fp8 + CPU text encoder + 1024×1024 is the intended config.
- First prompt is slow (CPU Qwen3-VL encode); generation speed is normal after.
- 1B/2B checkpoints run in bf16 directly on 8GB.

## Bonus: shard merge
`sefi_merge_transformer.py` streams the sharded transformer into ONE safetensors (<1GB RAM, bitwise identical — verified):
```
python_embeded\python.exe sefi_merge_transformer.py --src D:\sefi-image\transformer --dst D:\sefi-image\SeFi-5B-Base_transformer_bf16.safetensors
```

## License
Node pack + vendored inference code: MIT. **Model weights: CC BY-NC 4.0 (non-commercial)** — respect the SeFi-Image license.
