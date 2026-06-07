# coding=utf-8
import re
from pathlib import Path

from pydantic import BaseModel, Field

from archaeo.io.pdf import PdfDocSections, PdfDocSection, load_pdf_sections

CHARS_PER_TOKEN = 2.5


class Chunk(BaseModel):
    chunk_id: str

    text: str
    section_title: str | None = None

    metadata: dict = Field(default_factory=dict)


def pdf_section_to_text(section: PdfDocSection):
    parts = []
    if section.title:
        parts.append(f'# {section.title}')
    for block in section.blocks:
        if block.is_text():
            text = block.text.strip()
            if text:
                parts.append(text)
    return '\n\n'.join(parts)


def chunk_pdf_sections_by_chars(pdf_sections: PdfDocSections,
                                *,
                                max_chars: int=3000) -> list[Chunk]:

    chunks: list[Chunk] = []
    for section_id, section in enumerate(pdf_sections.sections):
        section_text = pdf_section_to_text(section)
        if not section_text:
            continue

        page_numbers = sorted({block.page_number for block in section.blocks})
        text_chunks = chunk_text_by_chars(section_text, max_chars=max_chars)

        for chunk_index, text_chunk in enumerate(text_chunks):
            # TODO: use hash?
            source_id = Path(pdf_sections.source).stem if pdf_sections.source else 'unknown'
            chunk_id = f'{source_id}_section_{section_id}_chunk_{chunk_index}'
            chunks.append(
                Chunk(chunk_id=chunk_id,
                      text=text_chunk,
                      section_title=section.title,
                      metadata={
                          'source': pdf_sections.source,
                          'section_index': section_id,
                          'chunk_index': chunk_index,
                          'section_level': section.level,
                          'page_start': min(page_numbers) if page_numbers else None,
                          'page_end': max(page_numbers) if page_numbers else None,
                          'parser': 'pymupdf_rule_based',
                          'quality': 'rough',
                      })
            )

    return chunks


def chunk_pdf_sections(pdf_sections: PdfDocSections,
                       *,
                       max_tokens: int=1000) -> list[Chunk]:
    max_chars = round(max_tokens * CHARS_PER_TOKEN)
    return chunk_pdf_sections_by_chars(pdf_sections, max_chars=max_chars)


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    # 尽量按中英文句末切
    parts = re.split(r'(?<=[。！？.!?])\s*', text)

    chunks = []
    current = []

    current_len = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            if current:
                chunks.append(''.join(current))
                current = []
                current_len = 0

            for i in range(0, len(part), max_chars):
                chunks.append(part[i:i + max_chars])
            continue

        if current and current_len + len(part) + 1 > max_chars:
            chunks.append(''.join(current))
            current = [part]
            current_len = len(part)
        else:
            current.append(part)
            current_len += len(part) + 1

    if current:
        chunks.append(''.join(current))

    return chunks


def chunk_text_by_chars(
    text: str,
    *,
    max_chars: int = 3000,
) -> list[str]:
    text = text.strip()

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r'\n\s*\n+', text)
        if p.strip()
    ]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0

            chunks.extend(split_long_text(para, max_chars))
            continue

        extra_len = len(para) + 2

        if current and current_len + extra_len > max_chars:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += extra_len

    if current:
        chunks.append('\n\n'.join(current))

    return chunks


def chunk_text(
    text: str,
    max_tokens: int = 1000,
) -> list[str]:
    max_chars = round(max_tokens * CHARS_PER_TOKEN)

    return chunk_text_by_chars(text, max_chars=max_chars)


def chunk_pdf(file_path: str,
              *,
              max_tokens: int=1000,
              n_pages: int | None=None) -> list[Chunk]:
    sections = load_pdf_sections(file_path, output_image_dir=None, n_pages=n_pages)
    chunks = chunk_pdf_sections(sections, max_tokens=max_tokens)
    return chunks


if __name__ == '__main__':
    # file = '/Users/andersc/data/dev/local_kb/TheEconomist.2026.06.06.pdf'
    file = '/Users/andersc/data/dev/local_kb/new_yorker.2026.06.08.pdf'
    file_chunks = chunk_pdf(file, max_tokens=3000, n_pages=50)
    for fc in file_chunks:
        print(fc)
        print()
