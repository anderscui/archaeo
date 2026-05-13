# coding=utf-8
from pathlib import Path

from archaeo.io.docs import Document
from archaeo.io.readers.markdown import read_markdown
from archaeo.io.readers.pdf import read_pdf


def read_document(file_path: str | Path,
                  *,
                  format: str | None = None,
                  backend: str | None = None) -> Document | None:
    file_path = Path(file_path)
    fmt = format or file_path.suffix.lower().lstrip('.')

    if fmt in {'pdf'}:
        return read_pdf(file_path)
    elif fmt in {'md', 'markdown'}:
        return read_markdown(file_path)
    else:
        raise ValueError(f'Unsupported file format: {fmt}')
