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

    pkg = types.ModuleType("dictionary_learning")
    pkg.__path__ = [str(_DL_ROOT / "dictionary_learning")]
    pkg.__package__ = "dictionary_learning"
    sys.modules["dictionary_learning"] = pkg

    for mod_name in [
        "dictionary_learning.buffer",
        "dictionary_learning.activault_s3_buffer",
    ]:
        sys.modules[mod_name] = types.ModuleType(mod_name)

_bootstrap()

from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.trainers.trainer import (
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
