# coding=utf-8
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from archaeo.io.pdf import BoundingBox


class DocumentMetadata(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)

    created_at: datetime | None = None
    updated_at: datetime | None = None

    source_format: Literal['pdf', 'epub', 'docx', 'md', 'txt']
    source_path: str | None = None

    # fallback
    extra: dict = Field(default_factory=dict)

    # extraction confidence
    confidence: float | None = None

    # raw pdf metadata
    raw: dict = Field(default_factory=dict)


class PageSpan(BaseModel):
    """Page index starts from 1."""
    start: int
    end: int

    @property
    def is_single_page(self):
        return self.start == self.end


class SourceLocation(BaseModel):
    page_span: PageSpan | None = None
    bbox: BoundingBox | None = None

    # For non-PDF formats.
    section_path: list[str] = Field(default_factory=list)
    block_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None

    @property
    def is_single_page(self) -> bool:
        return self.page_span is not None and self.page_span.is_single_page


class Block(BaseModel):
    type: Literal['title', 'paragraph', 'image', 'table', 'caption']
    text: str | None

    location: SourceLocation = Field(default_factory=SourceLocation)


class Section(BaseModel):
    title: str | None
    level: int | None
    blocks: list[Block] = Field(default_factory=list)


class Document(BaseModel):
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[Section] = Field(default_factory=list)
