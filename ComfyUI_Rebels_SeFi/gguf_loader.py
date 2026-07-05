# gguf_loader.py - minimal GGUF reader/dequantizer for Rebels SeFi.
# Supports the flat quant ladder we ship (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) plus
# F16/F32/BF16 passthrough. Dequantizes per-tensor at load time so the rest of
# the pipeline (fp8 storage cast, block swap) applies identically to GGUF and
# safetensors models.

import numpy as np
import torch


def _f16(buf):
    return buf.view("<f2").astype(np.float32)


def _deq_q8_0(raw, n):
    bs = 34
    b = raw.reshape(-1, bs)
    d = _f16(b[:, :2].copy())                       # (nb, 1)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _nibbles(qs):
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    return lo, hi


def _deq_q4_0(raw, n):
    bs = 18
    b = raw.reshape(-1, bs)
    d = _f16(b[:, :2].copy())
    lo, hi = _nibbles(b[:, 2:].copy())
    out = np.concatenate([(lo - 8.0) * d, (hi - 8.0) * d], axis=1)
    return out.reshape(-1)[:n]


def _deq_q4_1(raw, n):
    bs = 20
    b = raw.reshape(-1, bs)
    d = _f16(b[:, :2].copy())
    m = _f16(b[:, 2:4].copy())
    lo, hi = _nibbles(b[:, 4:].copy())
    out = np.concatenate([lo * d + m, hi * d + m], axis=1)
    return out.reshape(-1)[:n]


def _qh_bits(b, col):
    qh = b[:, col:col + 4].copy().view("<u4")       # (nb, 1)
    bits = ((qh >> np.arange(32, dtype=np.uint32)[None, :]) & 1).astype(np.float32)
    return bits                                      # (nb, 32)


def _deq_q5_0(raw, n):
    bs = 22
    b = raw.reshape(-1, bs)
    d = _f16(b[:, :2].copy())
    bits = _qh_bits(b, 2)
    lo, hi = _nibbles(b[:, 6:].copy())
    x0 = lo + bits[:, :16] * 16.0
    x1 = hi + bits[:, 16:] * 16.0
    out = np.concatenate([(x0 - 16.0) * d, (x1 - 16.0) * d], axis=1)
    return out.reshape(-1)[:n]


def _deq_q5_1(raw, n):
    bs = 24
    b = raw.reshape(-1, bs)
    d = _f16(b[:, :2].copy())
    m = _f16(b[:, 2:4].copy())
    bits = _qh_bits(b, 4)
    lo, hi = _nibbles(b[:, 8:].copy())
    x0 = lo + bits[:, :16] * 16.0
    x1 = hi + bits[:, 16:] * 16.0
    out = np.concatenate([x0 * d + m, x1 * d + m], axis=1)
    return out.reshape(-1)[:n]


# GGMLQuantizationType values (ggml constants)
_DEQUANT = {
    0: None,        # F32
    1: None,        # F16
    30: None,       # BF16
    2: _deq_q4_0,
    3: _deq_q4_1,
    6: _deq_q5_0,
    7: _deq_q5_1,
    8: _deq_q8_0,
}

_PASSTHROUGH_DTYPE = {0: np.float32, 1: np.float16}


def gguf_arch(path: str) -> str:
    import gguf
    r = gguf.GGUFReader(path)
    f = r.get_field("general.architecture")
    if f is None:
        return ""
    return bytes(f.parts[f.data[0]]).decode()


def iter_gguf_tensors(path: str):
    """Yield (name, torch_tensor_bf16, true_shape). Honors comfy.gguf.orig_shape."""
    import gguf
    r = gguf.GGUFReader(path)

    orig_shapes = {}
    for key, field in r.fields.items():
        if key.startswith("comfy.gguf.orig_shape."):
            name = key[len("comfy.gguf.orig_shape."):]
            orig_shapes[name] = tuple(int(field.parts[i][0]) for i in field.data)

    for t in r.tensors:
        qt = int(t.tensor_type)
        n = int(t.n_elements)
        shape = orig_shapes.get(t.name) or tuple(int(d) for d in reversed(t.shape))
        raw = np.array(t.data)  # copy out of the memmap
        if qt in (0, 1):
            arr = raw.view(_PASSTHROUGH_DTYPE[qt]).astype(np.float32)[:n]
            ten = torch.from_numpy(arr).reshape(shape).to(torch.bfloat16)
        elif qt == 30:  # BF16 raw bytes
            ten = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16).reshape(shape)
        elif qt in _DEQUANT and _DEQUANT[qt] is not None:
            arr = _DEQUANT[qt](raw.view(np.uint8), n)
            ten = torch.from_numpy(arr).reshape(shape).to(torch.bfloat16)
        else:
            raise ValueError(
                f"Tensor {t.name}: GGUF quant type {qt} not supported by this loader "
                "(flat quants Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 + floats only)."
            )
        yield t.name, ten, shape


def gguf_shapes(path: str) -> dict:
    """Header-only shape map (no dequant) for architecture derivation."""
    import gguf
    r = gguf.GGUFReader(path)
    orig = {}
    for key, field in r.fields.items():
        if key.startswith("comfy.gguf.orig_shape."):
            orig[key[len("comfy.gguf.orig_shape."):]] = tuple(int(field.parts[i][0]) for i in field.data)
    out = {}
    for t in r.tensors:
        out[t.name] = orig.get(t.name) or tuple(int(d) for d in reversed(t.shape))
    return out
