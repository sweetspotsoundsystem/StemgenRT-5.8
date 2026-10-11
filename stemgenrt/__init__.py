"""Current native training model and ONNX streaming inference APIs.

PyTorch is loaded only when a native-model API is requested.
"""
from importlib import import_module

__all__ = ["StreamingSeparator", "IndependentStems", "BandSeparator", "SpecialistSeparator",
           "StemgenRT58", "StreamingState", "render_scored_context"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = {"StreamingSeparator": ".streaming", "IndependentStems": ".independent",
              "BandSeparator": ".banded", "SpecialistSeparator": ".specialist"}.get(name, ".model")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
