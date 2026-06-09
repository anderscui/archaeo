# coding=utf-8
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from archaeo.chunking.base import Chunk, chunk_pdf
from archaeo.io.files import json_dump, json_load
from archaeo.llm_providers.base import BaseLlmProvider


TRANSLATE_CHUNK_PROMPT = """\
请将以下文本翻译为中文。

要求：
- 忠实于原文，不增删信息。
- 保留 Markdown 标题、列表、段落结构。
- 技术术语应准确、自然。
- 专有名词首次出现时可保留英文原文。
- 不要总结，不要解释，只输出译文。

原文如下：

{text}
"""


class TranslatedChunk(BaseModel):
    chunk_id: str

    source_text: str
    translated_text: str | None = None

    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslationResult(BaseModel):
    source: str | None = None
    target_lang: str = 'zh-CN'
    chunks: list[TranslatedChunk] = Field(default_factory=list)


def translate_text(
    text: str,
    provider: BaseLlmProvider,
    *,
    target_lang: str = 'zh-CN',
    prompt_template: str = TRANSLATE_CHUNK_PROMPT,
    **kwargs,
) -> str:
    prompt = prompt_template.format(
        text=text,
        target_lang=target_lang,
    )

    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]

    return provider.chat(
        messages,
        stream=False,
        think=False,
        **kwargs,
    )


def translate_chunks(
    chunks: list[Chunk],
    provider: BaseLlmProvider,
    *,
    target_lang: str = 'zh-CN',
    cache_file: str | Path | None = None,
    **kwargs,
) -> TranslationResult:
    result = TranslationResult(
        source=chunks[0].metadata.get('source') if chunks else None,
        target_lang=target_lang,
    )

    done: dict[str, TranslatedChunk] = {}

    if cache_file:
        cache_file = Path(cache_file)
        if cache_file.exists():
            cached = TranslationResult.model_validate(
                json_load(cache_file)
            )
            done = {
                chunk.chunk_id: chunk
                for chunk in cached.chunks
            }
            result = cached

    for chunk in tqdm(chunks, desc=f'Translating (by {provider.name})'):
        if chunk.chunk_id in done and done[chunk.chunk_id].translated_text:
            continue

        translated = translate_text(
            chunk.text,
            provider,
            target_lang=target_lang,
            **kwargs,
        )

        translated_chunk = TranslatedChunk(
            chunk_id=chunk.chunk_id,
            source_text=chunk.text,
            translated_text=translated,
            section_title=chunk.section_title,
            metadata=chunk.metadata,
        )

        # 如果之前已有同 id chunk，替换；否则追加
        result.chunks = [
            c for c in result.chunks
            if c.chunk_id != chunk.chunk_id
        ]
        result.chunks.append(translated_chunk)

        # 保持原 chunk 顺序
        order = {
            chunk.chunk_id: i
            for i, chunk in enumerate(chunks)
        }
        result.chunks.sort(
            key=lambda c: order.get(c.chunk_id, 10**9)
        )

        if cache_file:
            json_dump(
                result.model_dump(mode='json'),
                cache_file,
            )

    return result


def translate_pdf_by_chunks(
    file_path: str | Path,
    provider: BaseLlmProvider,
    *,
    target_lang: str = 'zh-CN',
    max_tokens: int = 1000,
    n_pages: int | None = None,
    cache_file: str | Path | None = None,
    **kwargs,
) -> TranslationResult:
    chunks = chunk_pdf(
        str(file_path),
        max_tokens=max_tokens,
        n_pages=n_pages,
    )

    return translate_chunks(
        chunks,
        provider,
        target_lang=target_lang,
        cache_file=cache_file,
        **kwargs,
    )


def translation_result_to_markdown(
    result: TranslationResult,
) -> str:
    parts = []
    for chunk in result.chunks:
        translated_text = (chunk.translated_text or '').strip()
        if translated_text:
            parts.append(translated_text)

    return '\n\n'.join(parts)


def save_translation_markdown(
    result: TranslationResult,
    output_file: str | Path
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    md = translation_result_to_markdown(
        result
    )

    output_file.write_text(md, encoding='utf-8')


def translation_result_to_bilingual_markdown(
    result: TranslationResult,
    *,
    source_heading: str = '原文',
    translation_heading: str = '译文',
    include_metadata: bool = False,
) -> str:
    parts = []

    for i, chunk in enumerate(result.chunks, start=1):
        # title = chunk.section_title or f'Chunk {i}'

        # parts.append(f'## {i}. {title}')
        # parts.append('')

        if include_metadata:
            parts.append(f'<!-- chunk_id: {chunk.chunk_id} -->')
            parts.append('')

        if source_heading:
            parts.append(f'### {source_heading}')
            parts.append('')
        parts.append(chunk.source_text.strip())
        parts.append('')

        if translation_heading:
            parts.append(f'### {translation_heading}')
            parts.append('')
        parts.append((chunk.translated_text or '').strip())
        parts.append('')

        parts.append('---')
        parts.append('')

    return '\n'.join(parts).strip()


def save_bilingual_markdown(
    result: TranslationResult,
    output_file: str | Path,
    *,
    include_metadata: bool = False,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    md = translation_result_to_bilingual_markdown(
        result,
        source_heading='',
        translation_heading='',
        include_metadata=include_metadata
    )

    output_file.write_text(md, encoding='utf-8')


if __name__ == '__main__':
    import time
    from archaeo.llm_providers import OllamaProvider, OpenRouterProvider, OpenRouterModels

    # 260608：
    # cloud：gemini_flash_lite_3_1 > minimax_m3 = deepseek_v4_pro > qwen3_7_plus
    # local：gemma4:26b > qwen3.5:9b

    # llm = OllamaProvider('gemma4:e2b')  # 1.1, 2.4, 10.2 -> 忽略
    # llm = OllamaProvider('gemma4:e2b-it-qat')  # 1.1, 2.4, 9.9 -> 忽略
    # llm = OllamaProvider('gemma4:12b')  # 0.65, 8.5, 42.0 -> 忽略
    # llm = OllamaProvider('qwen3.5:2b')  # 0.3, 2.5, 10.5 -> 忽略
    # llm = OllamaProvider('qwen3.5:4b')  # 0.5, 3.9, 17.9 -> 忽略
    # llm = OllamaProvider('qwen3.5:9b-nvfp4')  # 0.5, 5.3, 25.5 -> 忽略（速度提升很有限，依然很慢）
    # llm = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_2_5)  # 1.3, 2.2, 6.6 -> 忽略
    # llm = OpenRouterProvider(OpenRouterModels.gpt_5_4_nano)  # 3.2, 4.3, 12.5 -> 忽略

    # llm = OllamaProvider('qwen3.5:9b')  # 0.63, 5.9, 25.9 -> 基本忽略
    # llm = OpenRouterProvider(OpenRouterModels.gpt_5_4_mini)  # 1.4, 3.1, 12.5 -> 普通
    # llm = OpenRouterProvider(OpenRouterModels.deepseek_v4_flash)  # 3.3, 7.1, 17.1 -> 普通
    # llm = OpenRouterProvider(OpenRouterModels.qwen3_7_plus)  # 2.3, 7.6, 20.1 -> 普通

    # llm = OllamaProvider('qwen3.5:0.8b')  # 0.3, 1.5, 5.7
    llm = OllamaProvider('gemma4:26b')  # 0.5, 4.1, 19.5
    # llm = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_3_1)  # 1.7, 2.2, 5.7
    # llm = OpenRouterProvider(OpenRouterModels.deepseek_v4_pro)  # 3.7, 6.6, 26.2
    # llm = OpenRouterProvider(OpenRouterModels.minimax_m3)  # 3.3, 7.5, 34.7

    # start = time.time()
    # print(translate_text('hello, world', llm))
    # print(f'time elapsed: {time.time() - start}')
    #
    # text = """COCKTAIL PARTIES have seen worse pickup attempts. In 1989 Andrew
    # Wylie, a literary agent, set his sights on Philip Roth. (“Every time I turned
    # around, there was this guy,” the novelist later said. “I discovered what it was
    # like to be a pretty girl.”) When Mr Wylie made his move, he declared that he
    # could get Roth a three-book deal worth $2m.
    #
    # Later, over lunch, he formally wooed him. Roth was underappreciated by his
    # publisher, Mr Wylie asserted, and his potentially lucrative foreign rights
    # were being neglected. He could handle the rights to Roth’s 14 novels in 30
    # territories, which amounted to 420 contracts renegotiated every seven years.
    # Roth’s revenue could increase by up to 500%. It was an offer the writer--
    # who had a mighty reputation but a puny bank balance—could not refuse."""
    # start = time.time()
    # print(translate_text(text, llm))
    # print(f'time elapsed: {time.time() - start}')
    #
    # text = """The hidden tastemakers of the literary world
    # Literary agents are more important than readers give them credit for
    # June 4th 2026
    #
    # COCKTAIL PARTIES have seen worse pickup attempts. In 1989 Andrew Wylie, a literary agent, set his sights on Philip Roth. (“Every time I turned around, there was this guy,” the novelist later said. “I discovered what it was like to be a pretty girl.”) When Mr Wylie made his move, he declared that he could get Roth a three-book deal worth $2m.
    # Later, over lunch, he formally wooed him. Roth was underappreciated by his publisher, Mr Wylie asserted, and his potentially lucrative foreign rights were being neglected. He could handle the rights to Roth’s 14 novels in 30 territories, which amounted to 420 contracts renegotiated every seven years. Roth’s revenue could increase by up to 500%. It was an offer the writerwho had a mighty reputation but a puny bank balance—could not refuse. Mr Wylie is famous, infamous even, in the publishing world: he earned the nickname “the Jackal”, after persuading several authors to ditch their agents and join him. When he wanted to expand his network of contacts in China, he went so far as to call Henry Kissinger to suggest that he could become his agent and that Kissinger should write a book on China. That book went on to sell well, and Mr Wylie gained footing and a stronger Rolodex for future business there. Mr Wylie’s strategy of focusing on foreign rights and sales “not only changed the fortunes of the writers on his list, but the shape of American literature abroad”, writes Laura McGrath, an assistant professor of English at Temple University, in a new survey of literary agents.
    # Agents, Ms McGrath argues, have many functions. They are spotters, identifying promising writers. They are confidants, helping those writers to produce their best work. And they are negotiators, doing deals on behalf of authors and getting them more money so they can have more time to write.
    # That means agents are often controversial figures within the publishing world. In 1897 a British publisher described agents as “generally a parasite”. (They take a commission of around 15% of an author’s earnings.) They retain an aura of mystery. The literary agent’s job is to “be invisible”, Ms McGrath contends, working behind the scenes to create a star and thereby shape what gets written and what people read.
    # “Middlemen” is one of a pile of recent books offering a behind-the-scenes account of how literature comes to people’s shelves. Ms McGrath is a lively writer, eschewing academic jargon in favour of anecdotes. She heads to the world’s largest book fair in Frankfurt. She watches a group of agents evaluate the unsolicited manuscripts in the “slush pile”. (Every agent’s dream is to find a “Harry Potter”-like gem among the dross, as J.K. Rowling’s agent did.) She describes the evolution of the “publishing lunch”: Candida Donadio, who found Joseph Heller in the slush pile, once joked that she took three lunches in a day. (Today there are fewer stiff drinks and more talk about limp sales.)
    # Agents try to devise creative answers to marketing problems. For example, how do you sell a book by an unknown writer? Sterling Lord, Jack Kerouac’s agent, started a trend when he worked with publishers to build hype around “On the Road”; they advertised Kerouac as “the voice of a new age”. Agents ever since have hoped their debut novelists can make waves: first novels now account for between 15% and 25% of the fiction published every year. Many of these books fetch high advances that sales cannot recoup. Most American novelists now publish only one title.
    # Literary agents are never objective, and their personal preferences shape what is printed. Because most agents and editors live in New York, in 200022 more novels were set there than in the next 30 most populous American cities combined. And certainly the proliferation of agents—there are now more than 1,500 in America—has something to do with the proliferation of mediocre books. Many should have never have been printed. But what is striking, at a time when taste is increasingly shaped by algorithms, is just how much an individual agent’s judgment still matters."""
    #
    # start = time.time()
    # print(translate_text(text, llm))
    # print(f'time elapsed: {time.time() - start}')

    # text = """China’s delivery drivers are its most obvious underclass
    #
    # New rules aim to help but the economy will keep them down
    #
    # June 4th 2026
    #
    # CHINESE CITIES can look like a tableau of bold colours as delivery drivers on scooters—some yellow, orange or blue—zip through traffic bearing cargo to waiting customers. It is in the distant outskirts that the colours come to a rest. Amid the warrens of buildings, drivers rent cheap rooms, parking their scooters in the narrow streets and hanging their matching jackets from windows to air them out.
    #
    # One colour absent from their uniforms—each associated with a different delivery service—is gloomy grey. But speak to them as they return home, and that is the dominant hue. For years the delivery industry attracted strivers who knew they could earn more than in factory jobs or who wanted easy cash while awaiting better opportunities. These days, the realisation has set in for many: this may be as good as it gets, and it is worse than it used to be.
    #
    # “Whatever you do, nothing’s easy,” says Mr Wu, a driver for Taobao Flash, a delivery service. On his break after a long morning, he has taken off his orange jacket, revealing tattooed arms. He used to make seven yuan ($1) per order but that has fallen to four, in part because millions are chasing gig work as the wider economy slows. “After food and housing, there’s barely anything left.”
    #
    # Many couriers have gruelling schedules—working as many as 14 hours a day and rarely taking time off. Being late cuts into earnings, so they race to beat deadlines. Accidents are common. Mr Guan, in the blue uniform of Shansong, a parcel service, accepts danger with fatalism. “There is nothing more frightening than not being able to eat,” he says. “Every morning when I wake up, it’s not safety I think about. It’s about earning enough.”
    #
    # Chaguan spoke to these drivers in Yuxinzhuang, a commuter enclave in the north of Beijing. It achieved modest fame over the past month, sparked by a new documentary about migrant workers. The half-hour film, “2026 Chinese Delivery Drivers Survival Report”, was an unflinching look at their lives, following them from their cramped quarters in Yuxinzhuang through Beijing’s streets. The news website that hosted the documentary pulled it after a couple of days online. Whether because of official censorship or corporate pressure, it made for a Streisand effect with Chinese characteristics: the documentary ended up on YouTube, attracting viewers around the world.
    #
    # In the past, Chinese media and officials have not shied away from discussing the pressures and perils faced by delivery drivers. Estimated to number as many as 20m, they are simply too ubiquitous and too woven into daily life to be ignored, bringing people food, clothes, medicine and more. During covid-19 lockdowns, they were a lifeline for many. They have been the subjects of articles, films, podcasts and books; indeed, a book by a former courier was a publishing sensation in China in 2023. Xi Jinping himself met delivery drivers on the eve of Chinese New Year a few months ago, noting that cities could not function without them. Much of the commentary from state media conveys compassion, implicitly promising that the Communist Party is attuned to their challenges and will help them. At the street level, that is not the direction of travel. Many drivers point to the end of China’s covid restrictions in 2022 as the moment when things started to sour. The hoped-for economic rebound never materialised and consumer spending—the ultimate source of demand for their serviceshas stayed sluggish ever since.
    #
    # More controversial is the government’s role as regulator and enabler of persistent problems. Many concerns focus not on drivers’ wages but on their working conditions. Algorithms dictate order flows, pushing them to work faster. Delivery platforms almost always side with consumers in disputes. And drivers are typically contractors for third-party staffing firms, which allows big e-commerce platforms to avoid payments for medical insurance and pensions. This adds up to extreme hazards and scant buffers. Surveys of drivers indicate that roughly a third have been hurt on the job and only a fifth have insurance against workplace injuries.
    #
    # On paper, rules to protect delivery drivers, introduced by the government in 2021, appear to be on the mark. They guaranteed minimum wages and required platforms to make order-dispatching algorithms more humane, for example. In practice, officials have generally failed to implement these standards and punish violations. In April the Communist Party’s powerful Central Committee issued new labour rules for all gig workers, including delivery drivers. Yet these largely restate existing measures, fuelling scepticism about whether they will actually make a difference.
    #
    # Even if officials are stricter in enforcement, structural realities stack up against drivers. With a low barrier to entry, the delivery industry is a final refuge for those unable to find more stable jobs elsewhere, ensuring that wages will remain under pressure so long as the economy is weak. Most drivers come from poorer parts of the country and struggle to acquire residency rights where they work. That limits their access to local social services, from unemployment insurance to hospitals. They are vulnerable not just because e-commerce platforms exploit them but because the state has made them so. Drivers cannot engage in true collective bargaining. They are allowed to join unions but these all fall under the party, which is more committed to stamping out occasional protests than uniting workers as a social force. “Everyone knows that organising is useless,” says Mr Guan. After a bowl of noodles, he was heading back to the roads for an evening of work. Daytime deliveries had fallen well short of his target but he remained hopeful he would get there if he stayed out past midnight. “The point is not that it’s exhausting. There’s just no other way.”"""
    #
    # start = time.time()
    # print(translate_text(text, llm))
    # print(f'time elapsed: {time.time() - start}')

    start = time.time()
    # pdf_file = '/Users/andersc/Downloads/A Comprehensive Survey on Vector Database - Storage and Retrieval Technique, Challenge (2026.03).pdf'
    # pdf_file = '/Users/andersc/Downloads/AI research interviews (2025.08).pdf'
    pdf_file = '/Users/andersc/data/dev/local_kb/new_yorker.2026.06.08.pdf'
    result = translate_pdf_by_chunks(pdf_file, llm,
                                     max_tokens=2000,
                                     n_pages=100,
                                     cache_file='/Users/andersc/Downloads/new_yorker.2026.06.08.json')
    save_translation_markdown(result, output_file='/Users/andersc/Downloads/new_yorker.2026.06.08.md')
    # save_bilingual_markdown(result, output_file='/Users/andersc/Downloads/ai interview1-3.md')
    print(f'time elapsed: {time.time() - start}')
