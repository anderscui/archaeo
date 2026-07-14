# coding=utf-8
import json
import logging
import os
import requests

from archaeo.llm_providers import BaseLlmProvider, ProviderRegistry
from archaeo.schemas.llm_models import ModelInfo

logger = logging.getLogger(__name__)
ProviderRegistry.register("qwen", lambda **kwargs: QwenProvider(**kwargs))

# qwen3-max
# qwen-plus
# qwen-flash（取代了 qwen-turbo）
# qwen3-omni-flash：input：text、audio、image、video；output：text、audio
# qwen3-omni-flash-realtime
# qwen3-vl-plus；qwen3-vl-flash；
# qwen-vl-ocr：更专注于文档、表格、试题、手写体文字等类型图像的文字提取能力
# qwen-audio-turbo
# qwen3-coder-plus
# qwen-mt-plus；qwen-mt-flash；qwen-mt-lite；qwen-mt-turbo
# qwen-image-plus；qwen-image；
# qwen-image-edit-plus；qwen-image-edit；qwen-mt-image；
# 文生图 v2：wan2.2-t2i-plus；wan2.2-t2i-flash；
# 文生视频：wan2.2-t2v-plus；wan2.5-t2v-preview；
# 文本 embedding：text-embedding-v4（Qwen3-Embedding）
# 多模态 embedding：qwen2.5-vl-embedding；tongyi-embedding-vision-plus；tongyi-embedding-vision-flash；
# 意图理解：tongyi-intent-detect-v3
# deepseek-r1；deepseek-v3.1；deepseek-v3.2-exp；
# kimi-k2-thinking；Moonshot-Kimi-K2-Instruct
# glm-4.6；glm-4.5；glm-4.5-air；


class QwenProvider(BaseLlmProvider):
    def __init__(self, model='qwen-plus', api_key=None):
        super().__init__('qwen', model)

        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', '')
        self.base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        logger.debug(f'{self.name} model {self.model} initialized')

    @property
    def capabilities(self):
        return {'chat'}

    def chat(self, messages, stream=False, extra_headers: dict = None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json'
        }
        if extra_headers and isinstance(extra_headers, dict):
            headers.update(extra_headers)

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
            # "stream_options": {
            #     "include_usage": True
            # },
            # "enable_thinking": True,
            # "response_format": {
            #     "type": "json_object"
            # }
        }
        resp = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            # stream=stream,
            timeout=30
        )
        resp.raise_for_status()

        if stream:
            return self._stream_chat(resp)

        data = resp.json()
        return data['choices'][0]['message']['content']

    def _stream_chat(self, resp):
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue

            line = raw_line.strip()

            # keep-alive or comments
            if line.startswith(b':'):
                continue

            if line.startswith(b'data:'):
                line = line[len(b'data:'):].strip()

            # what's DONE is done
            if line in (b'[DONE]',):
                break

            try:
                data = json.loads(line.decode('utf-8'))
            except Exception as e:
                logger.error(f'[{self.model}] [stream] json decode error: {e}')
                print(line)
                continue

            if 'choices' not in data:
                continue

            choice = data['choices'][0]
            msg = choice.get('message')
            if msg and msg.get('content'):
                yield msg['content']
                continue

            # delta mode
            delta = choice.get('delta')
            if delta and delta.get('content'):
                yield delta['content']

    async def achat(self, messages, stream=False):
        raise NotImplementedError(f'Async {self.name} not implemented yet')

    def embed(self, text: str):
        raise NotImplementedError(f'Embedding not implemented for {self.name}')

    def list_models(self) -> list[ModelInfo]:
        """
        DashScope currently does NOT provide a public models listing API.
        We maintain a minimal static registry here.
        """
        def _parse_caps(model_id: str):
            caps = set()
            mid = model_id.lower()

            if 'vl' in mid or 'vision' in mid:
                caps.add('vision')

            if any(kw in mid for kw in ['reason', 'thinking']):
                caps.add('reasoning')

            # qwen models generally support tool calling / json
            caps.update({'tool'})

            # ---- chat fallback ----
            if not caps or caps <= {'vision'}:
                caps.add('chat')

            return sorted(caps)

        # ---- use a STATIC model list ----
        static_models = [
            {
                'id': 'qwen-flash',
                'name': 'Qwen3 Flash',
                'context_length': 1_000_000,
                'parameter_size': None,
                'description': '通义千问系列速度最快、成本极低的模型，适合简单任务。通义千问Flash采用灵活的阶梯定价，相比通义千问Turbo计费更合理。',
            },
            {
                'id': 'qwen-plus',
                'name': 'Qwen3 Plus',
                'context_length': 1_000_000,
                'parameter_size': None,
                'description': '能力均衡，推理效果、成本和速度介于通义千问Max和通义千问Flash之间，适合中等复杂任务。',
            },
            {
                'id': 'qwen3-max',
                'name': 'Qwen3 Max',
                'context_length': 262_144,
                'parameter_size': None,
                'description': '通义千问系列效果最好的模型，适合复杂、多步骤的任务。',
            },
            {
                'id': 'qwen-long',
                'name': 'Qwen Long',
                'context_length': 10_000_000,
                'parameter_size': None,
                'description': '通义千问系列上下文窗口最长，能力均衡且成本较低的模型，适合长文本分析、信息抽取、总结摘要和分类打标等任务。',
            },

            # Vision
            {
                'id': 'qwen3-vl-plus',
                'name': 'Qwen3 VL Plus',
                'context_length': 262_144,
                'parameter_size': None,
                'description': '通义千问VL是具有视觉（图像）理解能力的文本生成模型，不仅能进行OCR（图片文字识别），还能进一步总结和推理，例如从商品照片中提取属性，根据习题图进行解题等。',
                'input_modality': ['text', 'image'],
                'output_modality': ['text']
            },
            {
                'id': 'qwen3-vl-flash',
                'name': 'Qwen3 VL Flash',
                'context_length': 262_144,
                'parameter_size': None,
                'description': '通义千问VL是具有视觉（图像）理解能力的文本生成模型，不仅能进行OCR（图片文字识别），还能进一步总结和推理，例如从商品照片中提取属性，根据习题图进行解题等。',
                'input_modality': ['text', 'image'],
                'output_modality': ['text']
            },
            {
                'id': 'qwen-vl-ocr',
                'name': 'Qwen3 VL OCR',
                'context_length': 34_096,
                'parameter_size': None,
                'description': '通义千问OCR模型是专用于文字提取的模型。相较于通义千问VL模型，它更专注于文档、表格、试题、手写体文字等类型图像的文字提取能力。它能够识别多种语言，包括英语、法语、日语、韩语、德语、俄语和意大利语等。',
                'input_modality': ['text', 'image'],
                'output_modality': ['text']
            },

            # Audio ...
        ]

        models: list[ModelInfo] = []

        for m in static_models:
            model_id = m['id']
            input_mods = m.get('input_modality', ['text'])
            output_mods = m.get('output_modality', ['text'])
            modality = '+'.join(input_mods) + '->' + '+'.join(output_mods)

            caps = _parse_caps(model_id)

            models.append(ModelInfo(
                id=f'{self.name}:{model_id}',
                name=m['name'],
                description=m.get('name'),
                modified_at=None,

                family='qwen',

                parameter_size=m.get('parameter_size'),
                context_length=m.get('context'),
                modality=modality,
                input_modality=input_mods,
                output_modality=output_mods,
                tokenizer=None,
                disk_size=None,

                pricing=None,

                providers=[self.name],
                capabilities=caps,
                supported_parameters=['temperature', 'top_p']
            ))

        return models


if __name__ == '__main__':
    llm = QwenProvider()
    resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=False)
    print(resp)

    resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=True)
    for chunk in resp:
        print(chunk, end='')

    for m in llm.list_models():
        print(m)
        print()
