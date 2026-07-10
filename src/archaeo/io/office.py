# coding=utf-8
from pathlib import Path
from typing import Any

from docx import Document

from archaeo import logger
from archaeo.io.docs import LocalFileMetadata, TocItem, Toc
from archaeo.io.files import get_absolute_path


def get_docx_metadata(
    file_path: str | Path,
    max_outline_level: int = 2,
    preview_paragraphs: int = 20,
) -> LocalFileMetadata:
    try:
        file_path = get_absolute_path(file_path)
        doc = Document(str(file_path))

        props = doc.core_properties
        metadata: dict[str, Any] = {}

        for key in (
            "title",
            "author",
            "subject",
            "keywords",
            "comments",
            "category",
            "last_modified_by",
            "created",
            "modified",
        ):
            val = getattr(props, key, None)
            if val:
                metadata[key] = str(val)

        toc_items: list[TocItem] = []
        preview: list[str] = []

        for para in doc.paragraphs:
            text = (para.text or "").strip()
            if not text:
                continue

            if len(preview) < preview_paragraphs:
                preview.append(text)

            style_name = para.style.name if para.style else ""
            if not style_name.startswith("Heading"):
                continue

            try:
                level = int(style_name.replace("Heading", "").strip())
            except ValueError:
                continue

            if level <= max_outline_level:
                toc_items.append(
                    TocItem(
                        level=level,
                        title=text,
                        page_number=None,
                    )
                )

        return LocalFileMetadata(
            metadata=metadata,
            outline=Toc(items=toc_items),
            preview="\n".join(preview),
        )

    except Exception as e:
        logger.warning("get docx metadata failed: %s, file=%s", e, file_path)
        return LocalFileMetadata()
