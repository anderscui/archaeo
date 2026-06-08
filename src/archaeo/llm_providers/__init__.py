# coding=utf-8
from .base import BaseLlmProvider, Capability
from .registry import ProviderRegistry
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider, OpenRouterModels
from .deepseek import DeepSeekProvider
from .qwen import QwenProvider

KNOWN_PROVIDERS = ['ollama', 'openrouter', 'deepseek', 'qwen']
