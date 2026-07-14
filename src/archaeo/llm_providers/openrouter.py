# coding=utf-8
import json
import logging
import requests
import os

from archaeo.llm_providers import BaseLlmProvider, Capability, ProviderRegistry
from archaeo.schemas.llm_models import ModelInfo, PricingInfo

logger = logging.getLogger(__name__)
ProviderRegistry.register("openrouter", lambda **kwargs: OpenRouterProvider(**kwargs))


class OpenRouterProvider(BaseLlmProvider):
    def __init__(self, model='openai/gpt-4o-mini', api_key=None):
        super().__init__('openrouter', model)

        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY', '')
        self.base_url = 'https://openrouter.ai/api/v1'
        logger.debug(f'{self.name} model {self.model} initialized')

    @property
    def capabilities(self):
        return {'chat'}

    def chat(self, messages, stream=False, think: bool | None = None, extra_headers: dict | None=None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream' if stream else 'application/json'
        }
        if extra_headers is not None and isinstance(extra_headers, dict):
            headers.update(extra_headers)

        payload = {
            'model': self.model,
            'messages': messages,
            'stream': stream
        }
        if think is not None:
            payload['reasoning'] = {'enabled': think}

        resp = requests.post(
            f'{self.base_url}/chat/completions',
            headers=headers,
            json=payload,
            stream=stream,
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
            # SSE comments or keep-alive
            if line.startswith(b':'):
                continue

            # If SSE is used
            if line.startswith(b'data:'):
                line = line[len(b'data:'):].strip()

            if line == b'[DONE]':
                break

            try:
                data = json.loads(line.decode('utf-8'))
            except Exception as e:
                logger.error(f'[{self.model}] [stream] json decode error: {e}')
                continue

            if 'choices' in data:
                # not delta stream
                msg = data['choices'][0].get('message')
                if msg and msg.get('content'):
                    yield msg['content']
                    break

                # delta stream
                delta = data['choices'][0].get('delta')
                if delta and delta.get('content'):
                    yield delta['content']

    async def achat(self, messages, stream=False):
        raise NotImplementedError('Async OpenRouter not implemented yet')

    def embed_batch(self,
                    texts: list[str],
                    extra_headers: dict | None = None,
                    **kwargs) -> list[list[float]]:
        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding inputs must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs must not be empty")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if extra_headers is not None and isinstance(extra_headers, dict):
            headers.update(extra_headers)

        payload = {
            "model": self.model,
            "input": texts,
            **kwargs,
        }

        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        result = resp.json()
        data = result.get("data")

        if not isinstance(data, list):
            raise ValueError(
                f"invalid embedding response: missing data list, response={result!r}"
            )

        if len(data) != len(texts):
            raise ValueError(
                "embedding count mismatch: "
                f"expected={len(texts)}, actual={len(data)}"
            )

        try:
            data.sort(key=lambda item: item["index"])

            indices = [item["index"] for item in data]
            expected_indices = list(range(len(texts)))
            if indices != expected_indices:
                raise ValueError(
                    "invalid embedding indices: "
                    f"expected={expected_indices}, actual={indices}"
                )

            embeddings = [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"invalid embedding response data: {data!r}"
            ) from exc

        if any(not isinstance(embedding, list) for embedding in embeddings):
            raise ValueError("invalid embedding response: embedding must be a list")

        return embeddings

    def list_models(self) -> list[ModelInfo]:
        def _parse_pricing(pricing: dict[str, str]) -> PricingInfo:
            return PricingInfo(
                unit='1M',
                input=PricingInfo.norm_price_per_token(pricing.get('prompt')),
                output=PricingInfo.norm_price_per_token(pricing.get('completion')),
            )

        def _parse_caps(in_mods: list[str], out_mods: list[str], supported_params: list[str]):
            caps = set()

            # core modalities and capabilities
            if 'text' in in_mods and 'text' in out_mods:
                caps.add(Capability.CHAT)

            if 'image' in in_mods and 'text' in out_mods:
                caps.add(Capability.VISION)

            if 'text' in in_mods and 'image' in out_mods:
                caps.add(Capability.IMAGE)

            if 'text' in in_mods and 'video' in out_mods:
                caps.add(Capability.VIDEO)

            if 'text' in in_mods and 'audio' in out_mods:
                caps.add(Capability.AUDIO)

            if (
                    'audio' in in_mods
                    and ('text' in out_mods or 'transcription' in out_mods)
            ):
                caps.add(Capability.SPEECH_TO_TEXT)

            if (
                    'text' in in_mods
                    and ('speech' in out_mods or 'audio' in out_mods)
            ):
                caps.add(Capability.TEXT_TO_SPEECH)

            if 'embeddings' in out_mods:
                caps.add(Capability.EMBEDDING)

            if 'rerank' in out_mods:
                caps.add(Capability.RERANK)

            # advanced capabilities
            if 'tools' in supported_params:
                caps.add(Capability.TOOL)

            if 'structured_outputs' in supported_params or 'response_format' in supported_params:
                caps.add(Capability.STRUCTURED_OUTPUT)

            if 'reasoning' in supported_params or 'include_reasoning' in supported_params:
                caps.add(Capability.REASONING)

            if 'web_search_options' in supported_params:
                caps.add(Capability.WEB_SEARCH)

            return sorted(caps)

        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            params = {'output_modalities': 'all'}
            resp = requests.get(f'{self.base_url}/models', headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # print(json.dumps(data, indent=2, ensure_ascii=False))

            models = []
            for model in data.get('data', []):
                model_id = model['id']
                name = model['name']

                arch = model.get('architecture', {})
                input_mods = arch.get('input_modalities', [])
                output_mods = arch.get('output_modalities', [])
                modality = '+'.join(input_mods) + '->' + '+'.join(output_mods) if input_mods or output_mods else None
                all_mods = input_mods + output_mods

                supported_parameters = model.get('supported_parameters', [])

                caps = _parse_caps(input_mods, output_mods, supported_parameters)

                models.append(ModelInfo(
                    id=f'{self.name}:{model_id}',
                    name=name,
                    description=model.get('description'),
                    modified_at=model.get('created'),

                    family=model.get('family'),

                    parameter_size=model.get('parameter_size'),
                    context_length=model.get('context_length'),
                    modality=modality,
                    input_modality=input_mods,
                    output_modality=output_mods,
                    tokenizer=arch.get('tokenizer'),
                    disk_size=None,

                    pricing=_parse_pricing(model.get('pricing', {})),

                    providers=[self.name],
                    capabilities=caps,
                    supported_parameters=supported_parameters
                ))

            return models
        except Exception as e:
            logger.error(f'{self.name} list models failed: {e}')
            return []


class OpenRouterModels:
    gpt_audio = 'openai/gpt-audio'  # $2.5-10, 2026.01
    gpt_audio_mini = 'openai/gpt-audio-mini'  # $0.6-2.4, 2026.01
    gpt_5_6_luna_pro = 'openai/gpt-5.6-luna-pro'  # $1-6, 2026.07
    gpt_5_6_luna = 'openai/gpt-5.6-luna'  # $1-6, 2026.07
    gpt_5_6_terra_pro = 'openai/gpt-5.6-terra-pro'  # $2.5-15, 2026.07
    gpt_5_6_terra = 'openai/gpt-5.6-terra'  # $2.5-15, 2026.07
    gpt_5_6_sol_pro = 'openai/gpt-5.6-sol-pro'  # $5-30, 2026.07
    gpt_5_6_sol = 'openai/gpt-5.6-sol'  # $5-30, 2026.07
    gpt_5_5_pro = 'openai/gpt-5.5-pro'  # $30-180, 2026.04
    gpt_5_5 = 'openai/gpt-5.5'  # $5-30, 2026.04
    gpt_5_4_pro = 'openai/gpt-5.4-pro'  # $30-180, $10/K web search, 2026.03
    gpt_5_4 = 'openai/gpt-5.4'  # $2.5-15, $10/K web search, 2026.03
    gpt_5_4_mini = 'openai/gpt-5.4-mini'  # $0.75-4.5, $10/K web search, 2026.03
    gpt_5_4_nano = 'openai/gpt-5.4-nano'  # $0.2-1.25, $10/K web search, 2026.03
    gpt_5_3_chat = 'openai/gpt-5.3-chat'  # $1.75-14, $100/K web search, 2026.03
    gpt_5_3_codex = 'openai/gpt-5.3-codex'  # $1.75-14, $10/K web search, 2026.02
    # gpt_5_2_codex = 'openai/gpt-5.2-codex'  # $1.75-14, $10/K web search, 2026.01
    # gpt_5_2 = 'openai/gpt-5.2'  # $1.75-14, $10/K web search, 2025.12
    # gpt_5_2_chat = 'openai/gpt-5.2-chat'  # $1.75-14, $10/K web search, 2025.12, GPT-5.2 Instant
    # gpt_5_2_pro = 'openai/gpt-5.2-pro'  # $21-168, $10/K web search, 2025.12

    # gpt_5_1 = 'openai/gpt-5.1'  # $1.25-10, $10/K web search, 2025.11
    # gpt_5_1_chat = 'openai/gpt-5.1-chat'  # $1.25-10, $10/K web search, 2025.11
    # gpt_5_1_codex = 'openai/gpt-5.1-codex'  # $1.25-10, 2025.11

    gpt_5 = 'openai/gpt-5'  # coding  # $1.25-10, 2025.08
    gpt_5_mini = 'openai/gpt-5-mini'  # $0.25-2, 2025.08
    gpt_5_nano = 'openai/gpt-5-nano'  # $0.05-0.4, 2025.08
    gpt_5_pro = 'openai/gpt-5-pro'  # $15-120, 2025.10
    gpt_5_codex = 'openai/gpt-5-codex'  # $1.25-10, 2025.09
    gpt_4_1 = 'openai/gpt-4.1'  # $2-8, 2025.04
    gpt_4_1_mini = 'openai/gpt-4.1-mini'  # translation, $0.4-1.6, 2025.04
    gpt_4_1_nano = 'openai/gpt-4.1-nano'  # translation, $0.1-0.4, 2025.04
    gpt_4o = 'openai/gpt-4o-2024-11-20'  # $2.5-10, 2024.11
    gpt_4o_mini = 'openai/gpt-4o-mini'  # $0.15-0.6, 2024.07

    gpt_whisper_large_3_turbo = 'openai/whisper-large-v3-turbo'  # speech recognition, $0.04/hour, 2026.05
    gpt_whisper_large_3 = 'openai/whisper-large-v3'  # speech recognition, $0.111/hour, 2026.05
    gpt_whisper_1 = 'openai/whisper-1'  # speech recognition, $0.006/minute, 2026.04
    gpt_4o_transcribe_mini = 'openai/gpt-4o-mini-transcribe'  # speech-to-text, $1.25/M, 2026.05
    gpt_4o_transcribe = 'openai/gpt-4o-transcribe'  # speech-to-text, $2.5/M, 2026.04
    gpt_4o_audio = 'openai/gpt-4o-audio-preview'  # $2.5-10, 2025.08
    gpt_4o_mini_tts = 'openai/gpt-4o-mini-tts-2025-12-15'  # $0.6/M, 2026.04

    gpt_image_2 = 'openai/gpt-image-2'  # $8-8, 2026.06
    gpt_image_1_mini = 'openai/gpt-image-1-mini'  # $2.5-2.5, 2026.06
    gpt_image_1 = 'openai/gpt-image-1'  # $10-10, 2026.06
    gpt_5_4_image_2 = 'openai/gpt-5.4-image-2'  # $8-15, 2026.04
    gpt_5_image = 'openai/gpt-5-image'  # $10-10, $10/K web search, 2025.10
    gpt_5_image_mini = 'openai/gpt-5-image-mini'  # $2.50-2, $10/K web search, 2025.10

    gpt_oss_20b = 'openai/gpt-oss-20b'
    gpt_oss_120b = 'openai/gpt-oss-120b'
    # gpt_o1 = 'openai/o1'  # $15-60
    # gpt_o3 = 'openai/o3'  # $2-8
    # gpt_o3_mini = 'openai/o3-mini'  # $1.1-4.4
    # gpt_o4_mini = 'openai/o4-mini'  # $1.1-4.4

    # gpt_4o_search = 'openai/gpt-4o-search-preview'  # 2025.03
    # gpt_4o_mini_search = 'openai/gpt-4o-mini-search-preview'  # 2025.03

    gpt_o3_deep_research = 'openai/o3-deep-research'  # $10-40, 2025.10
    gpt_o4_mini_deep_research = 'openai/o4-mini-deep-research'  # $2-8, 2025.10

    # embeddings
    openai_emb_3_large = 'openai/text-embedding-3-large'  # $0.13
    openai_emb_3_small = 'openai/text-embedding-3-small'  # $0.02
    openai_emb_2_ada = 'openai/text-embedding-ada-002' # $0.10, legacy

    claude_fable_5 = 'anthropic/claude-fable-5'  # $10-50, 2026.06
    claude_opus_4_8_fast = 'anthropic/claude-opus-4.8-fast'  # $10-50, 2026.05
    claude_opus_4_8 = 'anthropic/claude-opus-4.8'  # $5-25, 2026.05
    claude_opus_4_7_fast = 'anthropic/claude-opus-4.7-fast'  # $30-150, 2026.05
    claude_opus_4_7 = 'anthropic/claude-opus-4.7'  # coding, $5-25, 2026.04
    claude_opus_4_6 = 'anthropic/claude-opus-4.6'  # coding, $5-25, 2026.02
    # claude_opus_4_5 = 'anthropic/claude-opus-4.5'  # coding, $5-25, 2025.11
    # claude_opus_4_1 = 'anthropic/claude-opus-4.1'  # coding, $15-75, 2025.08
    claude_opus_4 = 'anthropic/claude-opus-4'  # coding, $15-75, 2025.05
    claude_sonnet_5 = 'anthropic/claude-sonnet-5'  # $2-10, 2026.07
    claude_sonnet_4_6 = 'anthropic/claude-sonnet-4.6'  # coding, $3-15, 2026.02
    claude_sonnet_4_5 = 'anthropic/claude-sonnet-4.5'  # coding, $3-15, 2025.09
    claude_sonnet_4 = 'anthropic/claude-sonnet-4'  # image, coding, $3-15, 2025.05
    # claude_sonnet_3_7 = 'anthropic/claude-3.7-sonnet'  # 2025.02
    claude_haiku_4_5 = 'anthropic/claude-haiku-4.5'  # coding, $1-5, 2025.10
    # claude_haiku_3_5 = 'anthropic/claude-3.5-haiku'  # fastest model for daily tasks, 2024.11

    # google: translation, coding
    gemini_flash_3_5 = 'google/gemini-3.5-flash'  # mm(text, image, video, file, audio), $1.5-9, 2026.05
    gemini_flash_lite_3_1 = 'google/gemini-3.1-flash-lite'  # mm(text, image, video, file, audio), $0.25-1.5, 2026.05
    gemini_pro_3_1 = 'google/gemini-3.1-pro-preview'  # $2-12, 2026.02
    gemini_pro_3_1_tools = 'google/gemini-3.1-pro-preview-customtools'  # $2-12, 2026.02
    gemini_flash_3 = 'google/gemini-3-flash-preview'  # $0.5-3, 2025.12
    # gemini_flash_2_5 = 'google/gemini-2.5-flash'  # translation, $0.3-2.5, 2025.06
    # gemini_flash_lite_2_5 = 'google/gemini-2.5-flash-lite'  # translation, $0.1-0.4, 2025.07
    # gemini_pro_2_5 = 'google/gemini-2.5-pro'  # translation, $1.25-10, 2025.06

    gemini_chirp_3 = 'google/chirp-3'  # speech-to-text, $0.016/minute, 2026.05
    gemini_flash_tts_3_1 = 'google/gemini-3.1-flash-tts-preview'  # text-to-speech, $1-20, 2026.04
    gemini_pro_3_image = 'google/gemini-3-pro-image'  # $2-12, Nano Banana, 2026.06
    gemini_flash_3_1_image_lite = 'google/gemini-3.1-flash-lite-image'  # $0.25-1.5, Nano Banana, 2026.06
    gemini_flash_3_1_image = 'google/gemini-3.1-flash-image'  # $0.5-3, Nano Banana, 2026.06
    gemini_flash_2_5_image = 'google/gemini-2.5-flash-image'  # $0.3-2.5, Nano Banana, 2025.10

    google_veo_3_1_fast = 'google/veo-3.1-fast'  # video gen, $0.1 per sec, 2026.04
    google_veo_3_1_lite = 'google/veo-3.1-lite'  # video gen, $0.05 per sec, 2026.04
    google_veo_3_1 = 'google/veo-3.1'  # video gen, $0.4 per sec, 2026.03

    google_lyria_3_pro = 'google/lyria-3-pro-preview'  # music gen, $0.08 per song, 2026.03
    google_lyria_3_clip = 'google/lyria-3-clip-preview'  # music gen, $0.04 per song, 2026.03

    gemini_emb_1 = 'google/gemini-embedding-001'  # $0.15, 2025.10
    gemini_emb_2 = 'google/gemini-embedding-2'  # $0.2, 2026.05

    # gemma_4_31b = 'google/gemma-4-31b-it'
    # gemma_4_26b = 'google/gemma-4-26b-a4b-it'

    # qwen
    qwen3_7_plus = 'qwen/qwen3.7-plus'  # $0.4-1.6, 2026.06
    qwen3_7_max = 'qwen/qwen3.7-max'  # $2.5-7.5, 2026.05
    qwen3_6_max = 'qwen/qwen3.6-max-preview'  # $1.3-7.8, 2026.04
    qwen3_6_plus = 'qwen/qwen3.6-plus'  # $0.325-1.95, 2026.04
    qwen3_6_flash = 'qwen/qwen3.6-flash'  # 0.25-1.5, 2026.04
    qwen3_5_flash = 'qwen/qwen3.5-flash-02-23'  # 0.065-0.26, 2026.02
    qwen3_max_thinking = 'qwen/qwen3-max-thinking'  # $0.78-3.9, 2026.02
    qwen3_max = 'qwen/qwen3-max'  # $0.78-3.9, 2025.09
    qwen3_vl_235b_thinking = 'qwen/qwen3-vl-235b-a22b-thinking'  # 0.26-2.6, 2025.09
    qwen3_vl_235b_instruct = 'qwen/qwen3-vl-235b-a22b-instruct'  # 0.2-0.88, 2025.09
    # qwen3_vl_30b_instruct = 'qwen/qwen3-vl-30b-a3b-instruct'  # $0.13-0.52, 2025.10

    qwen3_coder_next = 'qwen/qwen3-coder-next'  # 80B-A3B, $0.14-0.8, 2026.02
    qwen3_coder_plus = 'qwen/qwen3-coder-plus'  # $0.65-3.25, 2025.09
    qwen3_coder = 'qwen/qwen3-coder'  # $0.22-1.8, 2025.07, Qwen3-Coder-480B-A35B-Instruct
    # qwen3_coder_flash = 'qwen/qwen3-coder-flash'  # coding, $0.195-0.975, 2025.09

    qwen3_emb_8b = 'qwen/qwen3-embedding-8b'  # $0.01
    qwen3_emb_4b = 'qwen/qwen3-embedding-4b'  # $0.02
    # qwen3_emb_06b = 'qwen/qwen3-embedding-0.6b'  # $0.01

    qwen3_asr_flash_2602 = 'qwen/qwen3-asr-flash-2026-02-10'  # audio -> transcription, $$0.000035/second, 2026.05

    deepseek_v4_flash = 'deepseek/deepseek-v4-flash'  # , $0.0983-0.1966, 2026.04
    deepseek_v4_pro = 'deepseek/deepseek-v4-pro'  # , $0.435-0.87, 2026.04
    deepseek_r1 = 'deepseek/deepseek-r1'  # 671b-37b, $0.7-2.5, 2025.01

    kimi_k2_7_code = 'moonshotai/kimi-k2.7-code'  # coding, $0.75-3.5 2026.06
    kimi_k2_6 = 'moonshotai/kimi-k2.6'  # coding, $0.75-3.5 2026.04
    kimi_k2_5 = 'moonshotai/kimi-k2.5'  # coding, $0.44-2 2026.01

    glm_5_2 = 'z-ai/glm-5.2'  # coding, $0.42-1.32, 2026.06
    glm_5_1 = 'z-ai/glm-5.1'  # coding, $1.05-3.5, 2026.04
    glm_5v_turbo = 'z-ai/glm-5v-turbo'  # mm agent, $1.2-4, 2026.04
    glm_5_turbo = 'z-ai/glm-5-turbo'  # $1.2-4, 2026.03
    glm_5 = 'z-ai/glm-5'  # $0.6-2.08, 2026.02
    # glm_4_7_flash = 'z-ai/glm-4.7-flash'  # $0.06-0.4, 2026.01
    # glm_4_7 = 'z-ai/glm-4.7'  # coding, $0.4-1.5, 2025.12
    # glm_4_6v = 'z-ai/glm-4.6v'  # vision, $0.3-0.9, 2025.12

    minimax_m3 = 'minimax/minimax-m3'  # $0.3-1.2, 2026.06
    minimax_m2_7 = 'minimax/minimax-m2.7'  # $0.3-1.2, coding, 2026.03
    minimax_m2_5 = 'minimax/minimax-m2.5'  # $0.3-1.2, coding, 2026.02
    minimax_m2_her = 'minimax/minimax-m2-her'  # dialog, $0.3-1.2, 2026.01
    minimax_m2 = 'minimax/minimax-m2'  # $0.255-1.0, coding, 2025.10

    grok_build_01 = 'x-ai/grok-build-0.1'  # coding, text/image -> text, $1-2, 2026.05
    grok_4_5 = 'x-ai/grok-4.5'  # $2-6, 2026.07.08
    grok_4_3 = 'x-ai/grok-4.3'  # $1.25-2.5, 2026.05
    grok_4_20_multi_agent = 'x-ai/grok-4.20-multi-agent'  # $2-6, 2026.03
    grok_4_20 = 'x-ai/grok-4.20'  # $2-6, 2026.03
    # grok_4_1_fast = 'x-ai/grok-4.1-fast'  # $0.2-0.5, 2025.11
    grok_4 = 'x-ai/grok-4'  # $3-15, 2025.07

    grok_voice_tts_1 = 'x-ai/grok-voice-tts-1.0'  # text -> audio, $15/M, 2026.05
    grok_img_img = 'x-ai/grok-imagine-image-quality'  # text/image -> image, $0.05/image, 2026.05
    grok_img_video = 'x-ai/grok-imagine-video'  # text/image -> video, $0.05/second, 2026.05

    mai_voice_2 = 'microsoft/mai-voice-2'  # tts, $22/M, 2026.06
    mai_transcribe_1_5 = 'microsoft/mai-transcribe-1.5'  # stt, $0.36/hour, 2026.06
    mai_image_2_5 = 'microsoft/mai-image-2.5'  # text/image -> image, $5/M tokens, 2026.06

    mistral_mini_transcribe = 'mistralai/voxtral-mini-transcribe'  # audio -> transcription, $$0.003/second, 2026.05

    pplx_emb_v1_06b = 'perplexity/pplx-embed-v1-0.6b'
    pplx_emb_v1_4b = 'perplexity/pplx-embed-v1-4b'


if __name__ == '__main__':
    import time
    from archaeo.io.files import json_dump

    # llm = OpenRouterProvider(OpenRouterModels.qwen3_6_plus)
    # resp = llm.chat(messages=[{'role': 'user', 'content': '请给我将一个关于程序员的笑话，用英语。'}], stream=False)
    # print(resp)

    # resp = llm.chat(messages=[{'role': 'user', 'content': 'hello，世界。'}], stream=True)
    # for chunk in resp:
    #     print(chunk, end='')

    # all_models = llm.list_models()
    # model_data = [model.model_dump(mode='json') for model in all_models]
    # json_dump(model_data, '~/Downloads/openrouter_model_info.json', indent=2)
    #
    # for m in llm.list_models()[:1000]:
    #     print(m)
    #     print()

    # llm = OpenRouterProvider(OpenRouterModels.pplx_emb_v1_06b)  # 10: 1.3 vs 13
    # llm = OpenRouterProvider(OpenRouterModels.pplx_emb_v1_4b)  # 10: 1.8 vs 13
    # llm = OpenRouterProvider(OpenRouterModels.openai_emb_3_small) # 10: 2.5 vs. 15
    llm = OpenRouterProvider(OpenRouterModels.openai_emb_3_large) # 10: 2.6 vs. 17
    # llm = OpenRouterProvider(OpenRouterModels.gemini_emb_2) # 10: 2.75 vs. 16
    # llm = OpenRouterProvider(OpenRouterModels.qwen3_emb_8b) # 10: 27 vs. 152
    start = time.time()
    emb = llm.embed('hello, world')
    # print(emb)

    print(f'time elapsed: {time.time() - start}')

    texts = ['写给非哲学家的哲学入门',
             '如何快速了解一个行业',
             'Naked Statistics',
             'Introducing Bertrand Russell',
             '契诃夫的一生 (伊莱娜·内米洛夫斯基, 2018)'] * 2
    start = time.time()
    embs = llm.embed_batch(texts)
    print(embs[0][:10])
    print(f'time elapsed: {time.time() - start}')

    start = time.time()
    embs = []
    for text in texts:
        embs.append(llm.embed(text))
    print(embs[0][:10])
    print(f'time elapsed: {time.time() - start}')
