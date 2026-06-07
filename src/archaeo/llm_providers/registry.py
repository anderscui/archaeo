# coding=utf-8
from typing import Any


class ProviderRegistry:
    _providers = {}

    @classmethod
    def register(cls, name: str, provider_cls: Any):
        cls._providers[name] = provider_cls

    @classmethod
    def get(cls, name: str='ollama', **kwargs):
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        return cls._providers[name](**kwargs)
