# core/formatting/registry.py

from typing import Callable, Dict

RESPONSE_RENDERERS: Dict[str, Callable] = {}


def register_renderer(name: str):
    def decorator(func: Callable):
        RESPONSE_RENDERERS[name] = func
        return func
    return decorator
