# coding=utf-8
import logging
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from archaeo.llm_providers import BaseLlmProvider
from archaeo.schemas.llm_models import ModelInfo

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLlmProvider):
    """
    OpenAI-compatible provider based on the official OpenAI Python SDK.

    Supports:
    - Responses API
    - Streaming text output
    - Async Responses API
    - Embeddings API
    - Custom OpenAI-compatible base URL
    """
    def __init__(
        self,
        model: str = "gpt-5.5",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ):
        super().__init__("openai", model)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Pass api_key or set OPENAI_API_KEY."
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        logger.debug(f"{self.name} model {self.model} initialized, base_url={self.base_url}")

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "embedding"}

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        think: bool | None = None,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """
        Generate a response using the Responses API.

        `messages` may contain roles such as:
        - system
        - developer
        - user
        - assistant
        """
        self._validate_messages(messages)

        payload = self._build_response_payload(
            messages=messages,
            think=think,
            kwargs=kwargs,
        )

        if stream:
            return self._stream_chat(payload)

        response = self.client.responses.create(**payload)
        return response.output_text

    def _stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        stream = self.client.responses.create(
            **payload,
            stream=True,
        )

        for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                if delta:
                    yield delta

            elif event.type == "response.failed":
                error = getattr(event.response, "error", None)
                raise RuntimeError(f"OpenAI response failed: {error}")

            elif event.type == "error":
                error = getattr(event, "error", event)
                raise RuntimeError(f"OpenAI stream error: {error}")

    async def achat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        think: bool | None = None,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        self._validate_messages(messages)

        payload = self._build_response_payload(
            messages=messages,
            think=think,
            kwargs=kwargs,
        )

        if stream:
            return self._astream_chat(payload)

        response = await self.async_client.responses.create(**payload)
        return response.output_text

    async def _astream_chat(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        stream = await self.async_client.responses.create(
            **payload,
            stream=True,
        )

        async for event in stream:
            if event.type == "response.output_text.delta":
                delta = event.delta
                if delta:
                    yield delta

            elif event.type == "response.failed":
                error = getattr(event.response, "error", None)
                raise RuntimeError(f"OpenAI response failed: {error}")

            elif event.type == "error":
                error = getattr(event, "error", event)
                raise RuntimeError(f"OpenAI stream error: {error}")

    def embed_batch(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        if not texts:
            return []

        self._validate_embedding_inputs(texts)

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs,
        )

        data = list(response.data)

        if len(data) != len(texts):
            raise ValueError(
                "embedding count mismatch: "
                f"expected={len(texts)}, actual={len(data)}"
            )

        data.sort(key=lambda item: item.index)

        indices = [item.index for item in data]
        expected_indices = list(range(len(texts)))

        if indices != expected_indices:
            raise ValueError(
                "invalid embedding indices: "
                f"expected={expected_indices}, actual={indices}"
            )

        embeddings = [item.embedding for item in data]

        if any(not isinstance(embedding, list) for embedding in embeddings):
            raise ValueError(
                "invalid embedding response: embedding must be a list"
            )

        return embeddings

    def list_models(self) -> list[ModelInfo]:
        try:
            page = self.client.models.list()
            print(f'has next page: {page.has_next_page()}')
            print(page.model_dump())

            models: list[ModelInfo] = []

            for model in page.data:
                print(model)
                print()
                model_id = model.id

                models.append(
                    ModelInfo(
                        id=f"{self.name}:{model_id}",
                        name=model_id,
                        description=None,
                        modified_at=getattr(model, "created", None),

                        family=None,

                        parameter_size=None,
                        context_length=None,
                        modality=None,
                        input_modality=[],
                        output_modality=[],
                        tokenizer=None,
                        disk_size=None,

                        pricing=None,

                        providers=[self.name],
                        capabilities=[],
                        supported_parameters=[],
                    )
                )

            return models

        except Exception as exc:
            logger.error("%s list models failed: %s", self.name, exc)
            return []

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.async_client.close()

    def _build_response_payload(
        self,
        messages: list[dict[str, str]],
        think: bool | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            **kwargs,
        }

        # OpenAI Responses API does not use:
        #
        #     reasoning={"enabled": True}
        #
        # For reasoning models it accepts reasoning configuration such as:
        #
        #     reasoning={"effort": "medium"}
        #
        # Only supply it when explicitly enabled, so ordinary models and
        # OpenAI-compatible proxy services are not given an unsupported field.
        if think is True and "reasoning" not in payload:
            payload["reasoning"] = {"effort": "medium"}

        return payload

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        if not messages:
            raise ValueError("messages must not be empty")

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"messages[{index}] must be a dict")

            role = message.get("role")
            content = message.get("content")

            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"messages[{index}].role must be a non-empty string"
                )

            if not isinstance(content, str):
                raise TypeError(
                    f"messages[{index}].content must be a string"
                )

            if not content.strip():
                raise ValueError(
                    f"messages[{index}].content must not be empty"
                )

    @staticmethod
    def _validate_embedding_inputs(texts: list[str]) -> None:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding inputs must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must not be empty")


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    llm = OpenAIProvider(model='gpt-5.6-luna')
    # resp = llm.chat(messages=[{'role': 'user', 'content': '请给我讲一个关于程序员的笑话，好玩儿一点的。'}], stream=False)
    # print(resp)

    resp = llm.chat(messages=[{'role': 'user', 'content': '从人类文明史的角度来看，如果一定要找出10本最重要的书，会有哪些？'}], stream=True)
    for part in resp:
        print(part, end='')

    # print(llm.generate('请给我推荐几本 self-help 的书籍'))

    # all_models = llm.list_models()
    # model_data = [model.model_dump(mode='json') for model in all_models]
    # for model in model_data:
    #     print(model)
    #     print()
