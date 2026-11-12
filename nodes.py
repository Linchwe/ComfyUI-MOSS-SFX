"""ComfyUI-MOSS-SFX

Thin ComfyUI nodes for OpenMOSS MOSS-SoundEffect v2.0 (Apache-2.0).

- Model code comes from the pip-installed `moss_soundeffect_v2` package (OpenMOSS, Apache-2.0).
  This file only calls that package and converts the result to ComfyUI AUDIO.
- Model folder: ComfyUI/models/moss_soundeffect_v2/MOSS-SoundEffect-v2.0
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys

import folder_paths

_MODEL_ROOT = os.path.join(folder_paths.models_dir, "moss_soundeffect_v2")
_NO_MODEL = "(model folder not found: models/moss_soundeffect_v2/MOSS-SoundEffect-v2.0)"

MODE_STANDARD = "standard (everything on GPU)"
MODE_LOW_VRAM = "low_vram (for 8GB: only the active part on GPU)"
DIT_BF16 = "bfloat16 (half the memory)"
DIT_FP16 = "float16 (RTX 20 series: no bfloat16 hardware)"
DIT_FP32 = "float32 (original)"

# Only one loaded pipeline is kept.
_CACHE = {"key": None, "pipe": None}


def _list_models():
    found = []
    if os.path.isdir(_MODEL_ROOT):
        for name in sorted(os.listdir(_MODEL_ROOT)):
            if os.path.isfile(os.path.join(_MODEL_ROOT, name, "model_index.json")):
                found.append(name)
    return found or [_NO_MODEL]


def _free_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _unload():
    _CACHE["key"] = None
    _CACHE["pipe"] = None
    _free_cuda()


def _free_comfy_models():
    """Unload models that ComfyUI itself keeps in VRAM (e.g. a Qwen3-VL text encoder used earlier).

    This node loads MOSS outside ComfyUI's memory manager, so ComfyUI does not know it must
    make room. On 8GB cards a cached model plus MOSS does not fit, so free them first.
    """
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
        print("[MOSS-SFX] Unloaded other ComfyUI models from VRAM", flush=True)
    except Exception as exc:  # Generation is still attempted if this fails
        print(f"[MOSS-SFX] could not unload other ComfyUI models (ignored): {exc}", flush=True)
    _free_cuda()


def _bf16_hardware():
    """True on RTX 30 series (sm_80) or newer, which compute bfloat16 natively."""
    try:
        import torch

        return torch.cuda.get_device_capability(0)[0] >= 8
    except Exception:
        return True


@contextlib.contextmanager
def _autocast_dtype(target):
    """The MOSS package always runs under torch.autocast(bfloat16).

    For float16 mode, swap only that bfloat16 request for float16 while generating.
    Other autocast calls (the VAE uses float32) are left as they are.
    """
    import torch

    if target is None or target == torch.bfloat16:
        yield
        return
    original = torch.autocast

    class _Autocast(original):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if dtype == torch.bfloat16:
                dtype = target
            super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    torch.autocast = _Autocast
    try:
        yield
    finally:
        torch.autocast = original


def _compute_dtype(cfg):
    import torch

    return torch.float16 if cfg.get("dit_dtype") == DIT_FP16 else torch.bfloat16


def _unwrap_compiled(fn):
    """Unwrap the torch.compile'd function (Windows has no triton, so compiling fails)."""
    seen = 0
    while seen < 8:
        inner = getattr(fn, "_torchdynamo_orig_callable", None) or getattr(fn, "__wrapped__", None)
        if inner is None or inner is fn:
            break
        fn = inner
        seen += 1
    return fn


def _disable_compile(pipe):
    try:
        import torch._dynamo as dynamo

        dynamo.config.disable = True
    except Exception:
        pass
    try:
        engine = pipe.engine
        engine.model_fn = _unwrap_compiled(engine.model_fn)
    except Exception as exc:  # Generation is still attempted if this fails
        print(f"[MOSS-SFX] failed to disable torch.compile (ignored): {exc}", flush=True)


def _fix_flash_attn():
    """Fix MOSS hanging when another node (e.g. SeedVR2) injected a fake flash_attn."""
    try:
        import importlib

        mod = importlib.import_module("moss_soundeffect_v2.diffsynth.models.wan_video_dit")
        fa = getattr(mod, "flash_attn", None)
        if getattr(mod, "FLASH_ATTN_2_AVAILABLE", False) and not callable(getattr(fa, "flash_attn_func", None)):
            mod.FLASH_ATTN_2_AVAILABLE = False
            print("[MOSS-SFX] Detected a fake flash_attn; it will not be used.", flush=True)
    except Exception:
        pass


def _cast_dit(dit, dtype):
    """Cast the DiT to bfloat16/float16 but keep complex RoPE buffers (freqs_cis_*) exactly as they were."""
    saved = {n: b.detach().clone() for n, b in dit.named_buffers() if b.is_complex()}
    dit.to(dtype)
    for name, buf in saved.items():
        owner = dit
        *path, leaf = name.split(".")
        for p in path:
            owner = getattr(owner, p)
        owner._buffers[leaf] = buf.to(owner._buffers[leaf].device)


def _check_rope(dit):
    """Safety net: if a RoPE buffer was turned into a real tensor, rebuild it with the same shape."""
    import torch
    from moss_soundeffect_v2.diffsynth.models import wan_video_dit as m

    for name, buf in list(dit.named_buffers()):
        leaf = name.split(".")[-1]
        if leaf.startswith("freqs_cis") and not buf.is_complex():
            rows, width = buf.shape[0], buf.shape[-1]
            fresh = m.precompute_freqs_cis(width * 2, end=rows)
            owner = dit
            *path, last = name.split(".")
            for p in path:
                owner = getattr(owner, p)
            owner._buffers[last] = fresh.to(buf.device)
            print(f"[MOSS-SFX] rebuilt RoPE buffer {name} {tuple(fresh.shape)}", flush=True)


def _load_pipeline(cfg):
    import torch

    _fix_flash_attn()
    try:
        from moss_soundeffect_v2 import MossSoundEffectPipeline
    except Exception as exc:
        raise RuntimeError(
            "Could not import the moss_soundeffect_v2 package. Install it with pip first (see the README).\n"
            f"Cause: {type(exc).__name__}: {exc}"
        )
    _fix_flash_attn()

    path = cfg["path"]
    low_vram = cfg["mode"] == MODE_LOW_VRAM
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA (NVIDIA GPU) not found.")
    gpu = torch.device("cuda:0")

    fp16 = cfg["dit_dtype"] == DIT_FP16
    load_dtype = torch.float16 if fp16 else torch.bfloat16
    if not fp16 and not _bf16_hardware():
        print(
            "[MOSS-SFX] This GPU has no bfloat16 hardware (RTX 20 series or older). "
            "If generation is very slow or fails, set dit_dtype to float16.",
            flush=True,
        )

    print(
        f"[MOSS-SFX] Loading model: {path} ({'low_vram' if low_vram else 'standard'}, "
        f"{'float16' if fp16 else 'bfloat16'})",
        flush=True,
    )
    # In low_vram mode everything loads on CPU first; only the active part is moved to GPU.
    pipe = MossSoundEffectPipeline.from_pretrained(
        path, torch_dtype=load_dtype, device=("cpu" if low_vram else gpu)
    )
    engine = pipe.engine

    if cfg["disable_compile"]:
        _disable_compile(pipe)

    if cfg["dit_dtype"] in (DIT_BF16, DIT_FP16):
        _cast_dit(engine.dit, load_dtype)
    if low_vram:
        engine.device = gpu
        engine.enable_cpu_offload()

    _check_rope(engine.dit)
    _free_cuda()
    print("[MOSS-SFX] Model ready", flush=True)
    return pipe


def _get_pipeline(cfg):
    key = (cfg["path"], cfg["mode"], cfg["dit_dtype"], cfg["disable_compile"])
    if _CACHE["key"] == key and _CACHE["pipe"] is not None:
        return _CACHE["pipe"]
    _unload()
    pipe = _load_pipeline(cfg)
    _CACHE["key"] = key
    _CACHE["pipe"] = pipe
    return pipe


def _progress(total):
    """Progress callback replacing tqdm: updates the ComfyUI progress bar and honors Cancel."""
    try:
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        bar = ProgressBar(int(total))
    except Exception:
        mm = None
        bar = None

    def wrap(iterable, *args, **kwargs):
        count = 0
        for item in iterable:
            if mm is not None:
                mm.throw_exception_if_processing_interrupted()
            yield item
            count += 1
            if bar is not None:
                bar.update_absolute(count, int(total))

    return wrap


class MossSfxLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_list_models(), {"tooltip": "Model folder inside ComfyUI/models/moss_soundeffect_v2/"}),
                "memory_mode": (
                    [MODE_LOW_VRAM, MODE_STANDARD],
                    {"default": MODE_LOW_VRAM, "tooltip": "Use low_vram for 8GB cards. standard also works with 12GB or more."},
                ),
                "dit_dtype": (
                    [DIT_BF16, DIT_FP16, DIT_FP32],
                    {
                        "default": DIT_BF16,
                        "tooltip": "bfloat16 for RTX 30 series or newer. float16 for RTX 20 series "
                        "(same memory as bfloat16). float32 uses about twice the memory.",
                    },
                ),
                "disable_torch_compile": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Keep on for Windows (torch.compile is not used)."},
                ),
            },
            "optional": {
                "manual_model_path": (
                    "STRING",
                    {"default": "", "tooltip": "Full folder path; only needed if the model is stored elsewhere."},
                ),
            },
        }

    RETURN_TYPES = ("MOSS_SFX_MODEL",)
    RETURN_NAMES = ("moss_model",)
    FUNCTION = "load"
    CATEGORY = "audio/MOSS SFX"
    DESCRIPTION = "Selects MOSS-SoundEffect v2.0 settings. The model itself loads on first generation."

    def load(self, model, memory_mode, dit_dtype, disable_torch_compile, manual_model_path=""):
        manual = str(manual_model_path or "").strip().strip('"')
        path = manual or os.path.join(_MODEL_ROOT, model)
        if not os.path.isfile(os.path.join(path, "model_index.json")):
            raise FileNotFoundError(
                f"Model folder not found: {path}\n"
                "ComfyUI\\models\\moss_soundeffect_v2\\MOSS-SoundEffect-v2.0 must contain model_index.json."
            )
        return (
            {
                "path": path,
                "mode": memory_mode,
                "dit_dtype": dit_dtype,
                "disable_compile": bool(disable_torch_compile),
            },
        )


class MossSfxGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moss_model": ("MOSS_SFX_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "A single gunshot, short and sharp, with a brief echo."},
                ),
                "negative_prompt": ("STRING", {"multiline": True, "default": "music, voice"}),
                "seconds": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 250, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "sigma_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "keep_loaded": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "If off, the model is unloaded from VRAM after generation (slower next run)."},
                ),
            },
            "optional": {
                "unload_other_models": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload models ComfyUI keeps in VRAM (e.g. Qwen3-VL used earlier) before "
                        "generating. Keep on for 8GB cards.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "audio/MOSS SFX"
    DESCRIPTION = "Generates a sound effect from an English/Chinese prompt (48kHz, up to 30 s)."

    def generate(self, moss_model, prompt, negative_prompt, seconds, steps, cfg_scale, sigma_shift, seed, keep_loaded,
                 unload_other_models=True):
        import torch

        text = str(prompt).strip()
        if not text:
            raise ValueError("The prompt is empty.")

        if unload_other_models:
            _free_comfy_models()

        pipe = _get_pipeline(moss_model)
        try:
            with _autocast_dtype(_compute_dtype(moss_model)):
                wave = pipe(
                    prompt=text,
                    seconds=float(seconds),
                    num_inference_steps=int(steps),
                    cfg_scale=float(cfg_scale),
                    sigma_shift=float(sigma_shift),
                    seed=int(seed),
                    negative_prompt=str(negative_prompt or ""),
                    progress_bar_cmd=_progress(steps),
                )
            wave = wave.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if wave.ndim == 2:
                wave = wave.unsqueeze(0)
            sample_rate = int(pipe.sample_rate)
        finally:
            if not keep_loaded:
                pipe = None
                _unload()
            else:
                _free_cuda()
        return ({"waveform": wave, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "MossSfxLoader": MossSfxLoader,
    "MossSfxGenerate": MossSfxGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MossSfxLoader": "MOSS SFX Loader",
    "MossSfxGenerate": "MOSS SFX Generate",
}
