from __future__ import annotations

import ctypes
import ctypes.util
import gc

import torch

_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
_malloc_trim = getattr(_libc, "malloc_trim", None)


def trim_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _malloc_trim is not None:
        _malloc_trim(0)
