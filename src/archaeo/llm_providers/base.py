# coding=utf-8
import math
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv

from archaeo.schemas.llm_models import ModelInfo

load_dotenv()


class ProviderError(Exception):
    pass


class UnsupportedCapabilityError(ProviderError):
    pass


class Capability:
    CHAT = 'chat'
    VISION = 'vision'

    IMAGE = 'image'
    AUDIO = 'audio'
    VIDEO = 'video'

    SPEECH_TO_TEXT = 'stt'
    TEXT_TO_SPEECH = 'tts'

    EMBEDDING = 'embedding'
    RERANK = 'rerank'

    TOOL = 'tool'
    STRUCTURED_OUTPUT = 'structured_output'
    REASONING = 'reasoning'
    WEB_SEARCH = 'web_search'
    # LOGPROBS = 'logprobs'


class BaseProvider(ABC):
    """Top-level abstract provider."""

    @property
    def capabilities(self) -> set[str]:
        return set()

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


class BaseLlmProvider(BaseProvider):
    """Abstract base provider for all LLM backends."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], stream: bool = False, think: bool | None = False, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def achat(self, messages: list[dict[str, str]], stream: bool = False) -> Any:
        raise NotImplementedError

    def generate(self, text: str, **kwargs) -> str:
        messages = [{"role": "user", "content": text}]
        return self.chat(messages, stream=False, **kwargs)

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def count_tokens(self, text: str, bytes_per_token: int = 4) -> int:
        """
        Estimate token count by UTF-8 length.
        Works for most LLMs (OpenAI, Anthropic, Gemini, Ollama models, etc.)

        :param text: Input text
        :param bytes_per_token: Average bytes per token (default 4)
        :return: Estimated token count
        """
        if not text:
            return 0

        length = len(text.encode("utf-8"))
        return math.ceil(length / bytes_per_token)

    def list_models(self) -> list[ModelInfo]:
        """List available models."""
        return []
