"""Vendored SeFi-Image inference core (MIT) - https://github.com/jmliu206/SeFi-Image
Adapted for ComfyUI by realrebelai: relative imports, no runtime/cli/distributed modules."""
from .config import load_config
from .registry import infer_model_spec, ModelSpec
from .checkpoints import resolve_config_path
from .runner import SEFIInferenceRunner, SEFIRunnerDirect
