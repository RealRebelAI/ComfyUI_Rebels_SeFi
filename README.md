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


## Nodes (Rebels → SeFi)
**Rebels SeFi Loader** — three dropdowns: transformer (diffusion_models), Qwen3-VL folder (text_encoders), VAE folder (vae). Scale, Base/Turbo family, and the semantic/texture channel split are derived automatically from the checkpoint.
- `weight_dtype`: **bf16** (~10.4GB, needs 12GB+). fp8 keeps embedders/proj_out/time-embeds in bf16 for quality. FP8 IS CURRENTLY NOT WORKING!
- `text_encoder_device`: **cpu** (default; Qwen3-VL encodes once per prompt, no VRAM cost) or cuda.
- `delta_t` / `timestep_shift_alpha`: −1 = model defaults. Advanced dials — Base/RL default alpha 0.3, Turbo 1.0.

**Rebels SeFi Sampler** — prompt, steps (0 = model default: 50 Base/RL, **4 Turbo**), guidance (−1 = default: 4.0 Base/RL, **1.0 Turbo**), size (multiples of 16), seed → IMAGE.

Turbo checkpoints only support 4/8/10 steps and guidance 1.0 (enforced).

## 8GB VRAM notes
- 5B + bf16 + CPU text encoder + 1024×1024 is the intended config.
- First prompt is slow (CPU Qwen3-VL encode); generation speed is normal after.
- 1B/2B checkpoints run in bf16 directly on 8GB.



## License
Node pack + vendored inference code: MIT. **Model weights: CC BY-NC 4.0 (non-commercial)** — respect the SeFi-Image license.
