# coding=utf-8
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from archaeo.io.docs import LocalFileMetadata, TocItem, Toc
from archaeo.io.files import get_absolute_path


def get_markdown_metadata(
    file_path: str | Path,
    max_outline_level: int = 2,
    preview_lines: int = 10,
) -> LocalFileMetadata:
    path = get_absolute_path(file_path)

    text = path.read_text(encoding="utf-8", errors="ignore")

    md = MarkdownIt()
    tokens = md.parse(text)

    toc_items: list[TocItem] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open":
            level = int(token.tag.lstrip("h"))

            if level <= max_outline_level:
                title = ""

                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    title = (tokens[i + 1].content or "").strip()

                if title:
                    toc_items.append(
                        TocItem(
                            level=level,
                            title=title,
                            page_number=None,
                        )
                    )

            i += 3
            continue

        i += 1

    preview = "\n".join(text.splitlines()[:preview_lines])

    metadata: dict[str, Any] = {
        "line_count": len(text.splitlines()),
        "char_count": len(text),
    }

    return LocalFileMetadata(
        metadata=metadata,
        outline=Toc(items=toc_items),
        preview=preview,
    )


if __name__ == '__main__':
    file = '~/github/writing/nlp/models/all-models.md'
    metadata = get_markdown_metadata(file, max_outline_level=2)
    for item in metadata.outline.items:
        print(item.title, item.level)
