# coding=utf-8
from pathlib import Path

from archaeo.io.docs import Document
from archaeo.io.pdf import get_pdf_metadata, PdfDocument


def read_pdf(file_path: str | Path) -> Document:
    metadata = get_pdf_metadata(file_path)
    print(metadata)
    if not metadata:
        metadata = {}
    doc_metadata = {
        'title': metadata.get('title'),
        'author': metadata.get('author'),

    }


if __name__ == '__main__':
    # file = '~/data/dev/local_kb/ThoughtWorks - Technology Radar 1269.pdf'
    file = '~/data/dev/local_kb/Qwen-Image-2.0 Technical Report - 2605.10730v1.pdf'
    read_pdf(file)
