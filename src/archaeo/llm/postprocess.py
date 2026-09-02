# coding=utf-8
import re


def strip_markdown_lang_wrappers(text):
    # strip wrappers like ```json, ```
    if text is None:
        return text
    text = text.strip()
    text = re.sub(r'^```\w*\s*', '', text)
    text = re.sub(r'```$', '', text)
    text = text.strip()
    return text


def extract_json(text) -> str | None:
    # extract json part within the first \{ and \} pair
    match = re.search(r'\{[\s\S]*}', text)
    return match.group(0) if match else None


if __name__ == '__main__':
    import json

    r0 = """
```json
{
    "job_title": "全栈AI工程师",
    "location": "上海",
    "work_experience": "5-10年",
    "core_skills": [
        "Python",
        "PyTorch",
        "Agent",
        "提示词工程",
        "RAG"
    ],
    "plus_skills": [
        "阿里云AI大模型产品"
    ],
    "responsibilities": [
        "负责Agent方案设计和落地实施，包括但不限于业务架构设计、智能体搭建、提示词工程、RAG和全链路优化等"
    ]
}
```
    """

    r = extract_json(r0)
    data = json.loads(r)
    print(data)
    assert 'job_title' in data

    r = strip_markdown_lang_wrappers(r0)
    # print(r)
    data = json.loads(r)
    print(data)
    assert 'job_title' in data
