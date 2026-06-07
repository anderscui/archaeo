# coding=utf-8
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class PricingInfo(BaseModel):
    unit: Literal['1K', '1M'] = '1M'
    input: Optional[float] = None
    output: Optional[float] = None
    currency: str = 'USD'

    @staticmethod
    def norm_price_per_token(price_per_token: str | None):
        def to_float(val: str, unit='1M') -> float:
            try:
                return float(val) * (1_000_000 if unit == '1M' else 1000)
            except Exception:
                return 0.0

        if price_per_token is None:
            price_per_token = '0'
        return to_float(price_per_token)


class ModelInfo(BaseModel):
    """
    Schema for various LLM models.
    """
    id: str                          # openai/gpt-4o-mini
    name: str                        # GPT-4o Mini
    description: Optional[str] = None
    modified_at: Optional[datetime] = None

    # Model family identifier normalized by Archaeo,
    # e.g. 'gpt-5', 'deepseek-v3', 'qwen3'
    family: Optional[str] = None

    parameter_size: Optional[str] = None  # '8B', '30B', etc.
    context_length: Optional[int] = None
    modality: Optional[str] = None
    input_modality: list[Literal['text', 'image', 'audio', 'video', 'file']] = Field(default_factory=lambda: [])
    output_modality: list[Literal['text', 'image', 'audio', 'video', 'transcription', 'speech', 'embeddings', 'rerank']] = Field(default_factory=lambda: [])
    tokenizer: Optional[str] = None
    disk_size: Optional[int] = None  # for local models only

    pricing: Optional[PricingInfo] = None

    providers: list[str]             # ['openai', 'openrouter']
    capabilities: list[Literal[
        'chat', 'vision',
        'image', 'audio', 'video',
        'stt', 'tts',
        'embedding', 'rerank',
        'tool', 'structured_output', 'reasoning', 'web_search',
    ]] = Field(default_factory=lambda: ['chat'])
    supported_parameters: list[str] = Field(default_factory=lambda: [])
