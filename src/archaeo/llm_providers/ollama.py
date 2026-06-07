# coding=utf-8
import json
import logging

import requests

from archaeo.llm_providers import BaseLlmProvider, ProviderRegistry
from archaeo.schemas.llm_models import ModelInfo

logger = logging.getLogger(__name__)
ProviderRegistry.register("ollama", lambda **kwargs: OllamaProvider(**kwargs))


class OllamaProvider(BaseLlmProvider):
    def __init__(self, model="gemma4:12b", host="http://localhost:11434"):
        super().__init__('ollama')

        self.model = model
        self.host = host.rstrip("/")
        logger.debug(f'{self.name} model {self.model} initialized')

    @property
    def capabilities(self):
        return {"chat", "embed"}

    def _request(self, endpoint, payload, stream=False):
        url = f"{self.host}/{endpoint}"
        resp = requests.post(url, json=payload, stream=stream)
        resp.raise_for_status()
        return resp

    def _stream_chat(self, resp):
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode())
            data_chunk = data.get('message', {}).get('content')
            if data_chunk:
                yield data_chunk

    def chat(self, messages, stream=False, **kwargs):
        payload = {"model": self.model, "messages": messages, "stream": stream}
        resp = self._request("api/chat", payload, stream=stream)
        if stream:
            return self._stream_chat(resp)
        else:
            return resp.json()["message"]["content"]

    async def achat(self, messages, stream=False):
        raise NotImplementedError("Async Ollama not implemented yet")

    def embed(self, text: str):
        payload = {"model": self.model, "prompt": text}
        resp = self._request("api/embeddings", payload)
        return resp.json().get("embedding", [])

    def list_models(self) -> list[ModelInfo]:
        def _check_capabilities(name: str):
            name = name.lower()
            embedding_keywords = ['embed', 'nomic-embed', 'all-minilm', 'bge', 'e5']
            if any(kw in name for kw in embedding_keywords):
                return ['embedding']

            vision_keywords = ['vision', '-vl', 'llava', 'ocr']
            if any(kw in name for kw in vision_keywords):
                return ['chat', 'vision']

            return ['chat']

        try:
            resp = requests.get(f'{self.host}/api/tags')
            resp.raise_for_status()
            data = resp.json()

            models = []
            for model in data['models']:
                caps = _check_capabilities(model['name'])
                if 'vision' in caps:
                    modality = ['text', 'image']
                else:
                    modality = ['text']

                models.append(ModelInfo(
                    id=f'{self.name}:{model["name"]}',
                    name=model['name'],
                    description=None,
                    modified_at=model['modified_at'],
                    family=model.get('details', {}).get('family'),

                    parameter_size=model.get('details', {}).get('parameter_size'),
                    context_length=None,
                    modality='+'.join(modality),

                    tokenizer=None,
                    disk_size=model['size'],

                    pricing=None,
                    providers=[self.name],
                    capabilities=caps,
                    supported_parameters=[]
                ))

            return models
        except Exception as e:
            logger.error(f'{self.name} list models failed: {e}')
            return []


if __name__ == '__main__':
    llm = OllamaProvider()
    resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=False)
    print(resp)

    # print(llm.generate('1+1=?'))

    # for chunk in llm.chat([{'role': 'user', 'content': 'Tell me which programming language is the best.'}], stream=True):
    #     print(chunk, end='')

    # import time
    # time.sleep(10)

    for m in llm.list_models():
        print(m)
        print()
