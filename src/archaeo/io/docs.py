# coding=utf-8
from datetime import datetime
from typing import Literal, Any

from pydantic import BaseModel, Field

from archaeo.maths import Rectangle


class BoundingBox(BaseModel):
    left: float | int
    top: float | int
    right: float | int
    bottom: float | int

    @classmethod
    def from_tuple(cls, values: tuple[float, float, float, float]):
        """
        elements order: (left, top, right, bottom)
        :param values:
        :return:
        """
        left, top, right, bottom = values
        return cls(left=left,
                   top=top,
                   right=right,
                   bottom=bottom)

    def to_tuple(self) -> tuple[float, float, float, float]:
        return self.left, self.top, self.right, self.bottom

    def resize(self, horizontal_ratio, vertical_ratio):
        self.left = self.left * horizontal_ratio
        self.top = self.top * vertical_ratio
        self.right = self.right * horizontal_ratio
        self.bottom = self.bottom * vertical_ratio

    def expand(self, by=1.0):
        return BoundingBox.from_tuple((self.left-by, self.top-by, self.right+by, self.bottom+by))

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def round(self, n=3):
        self.left = round(self.left, n)
        self.right = round(self.right, n)
        self.top = round(self.top, n)
        self.bottom = round(self.bottom, n)
        return self

    @staticmethod
    def merge(bboxes: list["BoundingBox"]):
        left = min(box.left for box in bboxes)
        top = min(box.top for box in bboxes)
        right = max(box.right for box in bboxes)
        bottom = max(box.bottom for box in bboxes)
        return BoundingBox.from_tuple((left, top, right, bottom))

    @staticmethod
    def are_intersected(b1, b2, threshold=10.0):
        intersection = BoundingBox.intersection_of(b1, b2)
        if not intersection:
            return False
        if intersection.width < threshold or intersection.height < threshold:
            return False
        return True

    @staticmethod
    def intersection_of(b1, b2):
        r1 = Rectangle.from_tuple(b1.to_tuple())
        r2 = Rectangle.from_tuple(b2.to_tuple())
        intersection = r1.intersection(r2)
        return intersection


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


class TocItem(BaseModel):
    level: int
    title: str
    page_number: int | None = None

    @classmethod
    def load(cls, data: list, engine='pymupdf'):
        if engine != 'pymupdf':
            raise ValueError(f'Unsupported engine: {engine}')

        valid_len = 4
        if len(data) != valid_len:
            raise ValueError(f'Unsupported value format: use a seq of length {valid_len}, got {len(data)}')

        level, title, page_number, dest = data
        title = title or ''
        title = title.strip()

        data = {
            'level': level,
            'title': title,
            'page_number': page_number,
        }
        return cls.model_validate(data)


class Toc(BaseModel):
    items: list[TocItem]

    @classmethod
    def load(cls, data):
        items = [TocItem.load(item) for item in data]
        items = sorted(items, key=lambda item: item.page_number)
        return cls(items=items)


class LocalFileMetadata(BaseModel):
    metadata: dict[str, Any] = Field(default_factory=dict)
    outline: Toc = Field(default_factory=lambda: Toc(items=[]))
    preview: str = ''

    def to_search_text(self) -> str:
        parts: list[str] = []

        # for value in self.metadata.values():
        #     if isinstance(value, list):
        #         parts.extend(str(v) for v in value if v)
        #     elif value:
        #         parts.append(str(value))

        parts.extend(item.title for item in self.outline.items if item.title)

        if self.preview:
            parts.append(self.preview)

        return "\n".join(parts)
