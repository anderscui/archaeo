# coding=utf-8
from collections import defaultdict
from pathlib import Path

import json
import os

from archaeo.llm.postprocess import strip_markdown_lang_wrappers
from archaeo.llm_providers import OllamaProvider
from archaeo.llm_providers.openrouter import OpenRouterProvider, OpenRouterModels
from archaeo.io.files import json_dump, list_files, get_absolute_path, copy_file


def parse_one_jd(provider, jd: str) -> dict:
    #     prompt_template = """
    # 你将会看到一段招聘 JD，请从中提取结构化信息，并严格输出 JSON。
    #
    # 注意：
    # 1、只依据 JD 文本内容提取信息，不要编造。
    # 2、若某字段缺失，请填 null 或 []。
    # 3、core_skills 与 plus_skills 必须按最细颗粒度拆分：一句话包含多个技能时必须拆成独立条目。
    # 4、在提取技能时，只保留“技能名称本体”，删除形容词与能力描述：
    #     * 删除前缀：必须 / 需要 / 熟练 / 精通 / 具备 / 至少掌握 / 熟悉 / 了解 / 良好 / 基本了解 / 有…能力 等
    #     * 删除后缀：能力 / 知识 / 经验 / 基础 / 习惯 / 能力强 / 能独立… 等
    #     * 保留核心技能名词，例如：Python、Go、FastAPI、PostgreSQL、消息队列、Docker、RAG、SFT、LangGraph 等。
    # 5、加分项定义（包含以下任一词的句子只进入 plus_skills）：加分项 / 优先 / 更好 / 若熟悉更佳。
    # 6、输出必须是严格的 JSON，不要任何解释或额外内容。
    #
    # 需要提取的字段如下：
    #
    # {{
    # "job_title": string | null,
    # "salary_range": string | null,
    # "location": string | null,
    # "work_experience": string | null,
    # "education": string | null,
    # "core_skills": string[],
    # "plus_skills": string[],
    # "responsibilities": string[]
    # }}
    #
    # 请分析以下 JD，并严格按规则提取：
    #
    # 【JD 开始】
    # {jd_text}
    # 【JD 结束】
    #
    # 只输出 JSON。
    #     """

    prompt_template = """
    你将会看到一段招聘 JD，请从中提取结构化信息，并只输出 JSON 对象。

    要求：
    1、不得编造内容。
    2、若字段缺失，用 null 或 []。
    3、core_skills / plus_skills：一句话多个技能必须拆分为多个条目。
    4、技能只保留技能本体，删除前后缀中的形容词、经验、能力、习惯等描述。
    5、若句子包含“加分项 / 优先 / 更好 / 若熟悉更佳”，该句技能全部归入 plus_skills。
    6、若 JD 中提到专业要求（如：计算机相关专业、软件工程、电子信息等），请提取到 major 字段；若无则填 null。
    7、最终输出必须是能直接 JSON.parse() / json.loads() 的 JSON 对象。

    字段结构：

    {{
    "job_title": string | null,
    "salary_range": string | null,
    "location": string | null,
    "work_experience": string | null,
    "education": string | null,
    "major": string[] | null,
    "core_skills": string[],
    "plus_skills": string[],
    "responsibilities": string[]
    }}

    请分析以下 JD：

    【JD 开始】
    {jd_text}
    【JD 结束】

    只输出 JSON 对象本身，不输出 Markdown、代码块或其他内容。
        """

    prompt = prompt_template.format(jd_text=jd)
    print(prompt)

    extra_headers = {
        'HTTP-Referer': 'https://anderscui.github.io/',
        'X-Title': 'archaeo'
    }

    result = provider.generate(prompt, extra_headers=extra_headers)
    # print(result)
    result = strip_markdown_lang_wrappers(result)
    data = json.loads(result)
    # print(json.dumps(data, ensure_ascii=False, indent=2))

    return data


def load_raw_jobs(source_dir: str | Path) -> list[Path]:
    return list(list_files(source_dir, excludes=lambda f: f.suffix.lower() not in ('.png', '.txt', '.md')))


def rename_raw_jobs(source_dir: str | Path, output_dir: str | Path) -> list[Path]:
    assert source_dir and output_dir

    source_dir = get_absolute_path(source_dir)
    output_dir = get_absolute_path(output_dir)
    assert source_dir != output_dir

    name_fmt = 'jd{num}{ext}'

    raw_jd_files = load_raw_jobs(source_dir)
    grouped_files = defaultdict(list)
    for jd_file in raw_jd_files:
        grouped_files[jd_file.suffix].append(jd_file)
    # print(grouped_files.keys())

    counted = 0
    for k, v in sorted(grouped_files.items()):
        print(k)
        range_start = counted+1
        range_end = counted + len(v)
        print(f'range: {range_start}-{range_end}')
        # print(v)
        counted += len(v)

        targets = []
        targets2 = []
        done = []
        for i in range(range_start, range_end+1):

            cur_file = source_dir / name_fmt.format(num=i, ext=k)
            if cur_file.is_file() and cur_file.exists():
                # print(f'copied: {cur_file}')
                done.append(cur_file)
                new_file = output_dir / cur_file.name
                copy_file(cur_file, new_file)
            else:
                targets.append(cur_file)
                targets2.append(i)
                # print(f'to copy: {cur_file}')

        if targets2:
            targets2 = list(reversed(targets2))
            # print(f'targets2: {targets2}')
            for ext_file in v:
                if ext_file in done:
                    continue
                i = targets2.pop()
                new_file = output_dir / name_fmt.format(num=i, ext=k)
                # print(f'{ext_file.name} -> {new_file.name}')
                copy_file(ext_file, new_file)

        print('\n')

    return []


def extract_jd_contents(raw_jds: list[Path], target_dir: str | Path) -> list[Path]:
    return []


def try_parse_one_jd():
    test_jd = """
        全栈AI工程师-K·薪

        上海
        5-10年
        本科

    岗位职责：
    负责线下算力资源交付方案设计和落地实施，解决部署过程中遇到的软硬件兼容性和性能问题。
    负责大模型交付方案设计和落地实施，解决部署过程中遇到的模型兼容性和性能问题。
    负责Agent方案设计和落地实施，包括但不限于业务架构设计、智能体搭建、提示词工程、RAG和全链路优化等。

    任职要求：
    泛计算机专业，本科及以上学历。
    精通Python，熟悉主流深度学习框架，如TensorFlow、PyTorch等。
    熟练掌握Linux、k8s、网络相关领域知识和运维手段，具备大模型运行环境搭建、网络问题排查、系统级问题诊断和解决能力。
    熟练使用主流智能体开发平台开发智能体和工作流，有知识库和其他AI场景落地经验，具备包括智能体搭建与调优、MCP调用、工具调用与优化、RAG召回策略优化等能力。
    熟悉vLLM和SGLang在内的主流推理框架，具备一定的模型推理优化经验。
    熟悉阿里云AI大模型产品如PAI、百炼、点金或灵码，持有阿里云大模型ACP认证证书者优先
        """
    # main()

    # provider = OpenRouterProvider(OpenRouterModels.gpt_5_4_nano)
    # provider = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_3_5)
    # provider = OpenRouterProvider(OpenRouterModels.kimi_k3)
    # provider = OpenRouterProvider(OpenRouterModels.qwen3_7_plus)
    provider = OpenRouterProvider(OpenRouterModels.qwen3_8_flash)
    # provider = OpenRouterProvider(OpenRouterModels.deepseek_v4_flash_0731)
    # provider = OllamaProvider(model='gemma4:26b')
    parsed = parse_one_jd(provider, test_jd)
    print(parsed)


def main():
    jd_file = os.getenv('SE_JOB_FILE')
    jd_parsed_file = os.getenv('SE_JOB_PARSED_FILE')

    print(f'jd file: {jd_file}, target: {jd_parsed_file}')

    with open(jd_file, 'r') as f:
        jds_text = f.read()

    jds = [jd for jd in jds_text.split('=====') if jd.strip()]
    print(f'jd count: {len(jds)}')

    provider = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_3_5)
    parsed_jds = []
    for jd in jds:
        try:
            result = parse_one_jd(provider, jd)
            assert isinstance(result, dict)
        except Exception as e:
            print(f'parse jd error: {e}')
            result = {}

        parsed = {'jd': jd}
        parsed.update(result)
        parsed_jds.append(parsed)
        json_dump(parsed_jds, jd_parsed_file, indent=2)


if __name__ == "__main__":
    # try_parse_one_jd()

    raw_jd_dir = '~/Downloads/jobs/py-202608'
    renamed_jd_dir = '~/Downloads/jobs/py-202608-renamed'
    # raw_jd_files = load_raw_jobs(raw_jd_dir)
    # print(len(raw_jd_files))
    # print(raw_jd_files)
    rename_raw_jobs(raw_jd_dir, renamed_jd_dir)
