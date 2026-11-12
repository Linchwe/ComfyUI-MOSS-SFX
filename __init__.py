from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .cuesheet import MossSfxCuePrompt, MossSfxCueSheet

NODE_CLASS_MAPPINGS["MossSfxCueSheet"] = MossSfxCueSheet
NODE_CLASS_MAPPINGS["MossSfxCuePrompt"] = MossSfxCuePrompt
NODE_DISPLAY_NAME_MAPPINGS["MossSfxCueSheet"] = "MOSS SFX Cue Sheet"
NODE_DISPLAY_NAME_MAPPINGS["MossSfxCuePrompt"] = "MOSS SFX Cue Prompt (for Qwen3-VL)"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
