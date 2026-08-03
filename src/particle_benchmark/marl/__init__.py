"""MARL algorithms, imported lazily so representation utilities stay CPU-light."""

from __future__ import annotations

from importlib import import_module

__all__ = ["IPPO", "MAPPO", "MADDPG", "COMA", "CommNet", "VDN"]

_MODULES = {
    "IPPO": ".ippo",
    "MAPPO": ".mappo",
    "MADDPG": ".maddpg",
    "COMA": ".coma",
    "CommNet": ".commnet",
    "VDN": ".vdn",
}


def __getattr__(name: str) -> object:
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
