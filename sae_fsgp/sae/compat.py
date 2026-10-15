"""
Shim that lets us import dictionary_learning.trainers.top_k without
pulling in nnsight / aiohttp / boto3 (which are LLM-specific deps we
don't need for offline ResNet activation training).

Usage:
    from sae.compat import AutoEncoderTopK
"""
import sys
import types
from pathlib import Path

_DL_ROOT = Path(__file__).parent.parent.parent / "dictionary_learning"


def _bootstrap():
    if "dictionary_learning" in sys.modules:
        return

    dl_path = str(_DL_ROOT)
    if dl_path not in sys.path:
        sys.path.insert(0, dl_path)

    # Create a minimal package stub so relative imports inside the
    # trainers/ sub-package resolve without the heavy top-level __init__.
    pkg = types.ModuleType("dictionary_learning")
    pkg.__path__ = [str(_DL_ROOT / "dictionary_learning")]
    pkg.__package__ = "dictionary_learning"
    sys.modules["dictionary_learning"] = pkg

    # Stub out heavy modules that nothing in trainers/ actually calls
    for mod_name in [
        "dictionary_learning.buffer",
        "dictionary_learning.activault_s3_buffer",
    ]:
        sys.modules[mod_name] = types.ModuleType(mod_name)


_bootstrap()

from dictionary_learning.trainers.top_k import AutoEncoderTopK  # noqa: E402
from dictionary_learning.trainers.trainer import (  # noqa: E402
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    get_lr_schedule,
)

__all__ = [
    "AutoEncoderTopK",
    "set_decoder_norm_to_unit_norm",
    "remove_gradient_parallel_to_decoder_directions",
    "get_lr_schedule",
]
