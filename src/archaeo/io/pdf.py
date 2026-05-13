# coding=utf-8
import logging
import re
from pathlib import Path

import fitz
from pydantic import BaseModel, PrivateAttr, Field

from archaeo.iterable import rename_keys
from archaeo.maths import Rectangle

logger = logging.getLogger(__name__)

FLAG_SUPERSCRIPT = 1 << 0
FLAG_ITALIC = 1 << 1
FLAG_SERIFED = 1 << 2
FLAG_MONOSPACED = 1 << 3
FLAG_BOLD = 1 << 4


def save_image_block(
        image_block: dict,
        output_dir: Path,
        page_index: int,
        block_index: int,
        min_width: float = 30.0,
        min_height: float = 30.0
) -> Path | None:

    x0, y0, x1, y1 = image_block['bbox']
    width = x1 - x0
    height = y1 - y0
    if width < min_width or height < min_height:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    image_data = image_block['image']
    image_ext = image_block['ext']

    image_path = output_dir / f'page_{page_index}_img_{block_index}.{image_ext}'
    with open(image_path, 'wb') as f:
        f.write(image_data)

    return image_path


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


class PdfFont(BaseModel):
    font_name: str
    font_size: float
    font_color: int

    is_bold: bool
    is_italic: bool
    is_monospaced: bool


class PdfSpan(BaseModel):
    _font: PdfFont | None = PrivateAttr(default=None)

    page_number: int
    origin: tuple[float, float]
    bbox: BoundingBox

    text: str

    font_name: str
    font_size: float
    font_color: int
    ascender: float
    descender: float

    flags: int

    # not required in most cases
    chars: list = Field(default_factory=list)

    @classmethod
    def load(cls, data: dict, page_number):
        data = rename_keys(data, {
            'font': 'font_name',
            'size': 'font_size',
            'color': 'font_color'
        })
        data['page_number'] = page_number
        # data['origin'] = data['origin']
        data['bbox'] = BoundingBox.from_tuple(data['bbox'])

        return cls.model_validate(data)

    @property
    def font(self):
        if self._font is None:
            self._font = PdfFont(font_name=self.font_name,
                           font_size=self.font_size,
                           font_color=self.font_color,
                           is_bold=self.is_bold(),
                           is_italic=self.is_italic(),
                           is_monospaced=self.is_monospaced())
        return self._font

    def is_super_script(self):
        return (self.flags & FLAG_SUPERSCRIPT) > 0

    def is_italic(self):
        return (self.flags & FLAG_ITALIC) > 0

    def is_serifed(self):
        return (self.flags & FLAG_SERIFED) > 0

    def is_monospaced(self):
        return (self.flags & FLAG_MONOSPACED) > 0

    def is_bold(self):
        return (self.flags & FLAG_BOLD) > 0


class PdfLine(BaseModel):
    _text: str = PrivateAttr(default=None)

    page_number: int
    writing_mode: int
    dir: tuple[float, float] = ()
    bbox: BoundingBox

    spans: list[PdfSpan]

    object_type: str = 'line'

    @classmethod
    def load(cls, data: dict, page_number):
        data = rename_keys(data, {
            'wmode': 'writing_mode',
        })
        data['page_number'] = page_number
        # data['dir'] = Point.from_tuple(data['dir'])
        # data['bbox'] = BoundingBox.from_tuple(data['bbox'])

        spans = data.get('spans') or []
        if not spans:
            return None

        spans = [PdfSpan.load(span, page_number) for span in spans]
        spans = [span for span in spans if span is not None]
        if not spans:
            return None

        data['spans'] = spans
        span_bboxes = [span.bbox for span in spans]
        data['bbox'] = BoundingBox.merge(span_bboxes)
        return cls.model_validate(data)

    @property
    def text(self):
        if self._text is None:
            ends = []
            for i, s in enumerate(self.spans):
                if i < len(self.spans) - 1:
                    next_span = self.spans[i + 1]
                    end = '' if next_span.bbox.left - s.bbox.right < 0.1 else ' '
                    ends.append(end)
                else:
                    ends.append('')

            self._text = ''.join([s.text + end for s, end in zip(self.spans, ends)])
        return self._text

    # @property
    # def height(self):
    #     return self.bbox[3] - self.bbox[1]
    #
    # @property
    # def y_center(self):
    #     return (self.bbox[1] + self.bbox[3]) / 2


class PdfBlock(BaseModel):
    _text: str = PrivateAttr(default=None)

    page_number: int

    type: int
    block_number: int
    bbox: BoundingBox

    lines: list[PdfLine] = Field(default_factory=list)

    object_type: str = 'block'

    @classmethod
    def load(cls, data: dict, page_number, **kwargs):
        _type = data['type']
        if _type == 0:
            return TextBlock.load(data, page_number, **kwargs)
        elif _type == 1:
            return ImageBlock.load(data, page_number, **kwargs)
        return None

    def is_text(self):
        return self.type == 0

    def is_image(self):
        return self.type == 1

    @property
    def text(self) -> str | None:
        return None

    @property
    def is_single_line(self):
        return len(self.lines) == 1


class TextBlock(PdfBlock):
    type: int = 0

    flags: int

    @classmethod
    def load(cls, data: dict, page_number, **kwargs):
        data['block_number'] = data.pop('number', None)
        data['page_number'] = page_number
        data['bbox'] = BoundingBox.from_tuple(data['bbox'])

        lines = data.get('lines') or []
        if not lines:
            return None

        lines = [PdfLine.load(line, page_number) for line in lines]
        lines = [line for line in lines if line is not None]
        if not lines:
            return None

        data['lines'] = lines
        return cls.model_validate(data)

    @property
    def text(self):
        if self._text is None:
            self._text = '\n'.join([line.text for line in self.lines])
        return self._text


class ImageBlock(PdfBlock):
    type: int = 1

    # width and height of original image.
    width: int
    height: int

    ext: str
    image: bytes | None = Field(default=None, exclude=True)
    image_path: Path | None = None
    mask: bytes | None = None


    @classmethod
    def load(cls, data: dict, page_number: int, *, image_dir: Path | None = None, keep_in_memory: bool = False, **kwargs):
        data['block_number'] = data.pop('number', None)
        data['page_number'] = page_number

        if image_dir:
            saved_path = save_image_block(data, image_dir, page_number, data['block_number'])
            if not saved_path:
                return None
            data['image_path'] = saved_path

        if not keep_in_memory:
            data.pop('image', None)
            data.pop('mask', None)

        data['bbox'] = BoundingBox.from_tuple(data['bbox'])
        return cls.model_validate(data)

    @property
    def size(self):
        if self.image is not None:
            return len(self.image)
        if self.image_path and self.image_path.exists():
            return self.image_path.stat().st_size
        return 0

    # @property
    # def persisted(self):
    #     return self.image_path is not None


class PdfPage(BaseModel):
    page_number: int

    height: float
    width: float
    blocks: list[PdfBlock]

    object_type: str = 'page'

    @classmethod
    def load(cls, data: dict, page_number, image_dir: Path | None = None):
        data['page_number'] = page_number
        blocks = data.get('blocks') or []
        if not blocks:
            return None

        blocks = [PdfBlock.load(block, page_number, image_dir=image_dir) for block in blocks]
        blocks = [block for block in blocks if block is not None]
        data['blocks'] = blocks
        return cls.model_validate(data)


class PdfDocument(BaseModel):
    pages: list[PdfPage]

    object_type: str = 'document'

    @classmethod
    def load_file(cls,
                  file_path: str,
                  image_dir: str | Path | None = None,
                  n_pages: int = None):

        if n_pages is not None and n_pages < 1:
            raise ValueError(f'Number of pages should be a positive integer.')

        if image_dir:
            image_dir = Path(image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)

        pages = []
        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    page_number = page.number + 1
                    if n_pages is not None and page_number > n_pages:
                        break
                    page_obj = page.get_text('dict')
                    page_loaded = PdfPage.load(page_obj, page_number, image_dir)
                    if page_loaded is None:
                        continue
                    pages.append(page_loaded)
        except Exception:
            logger.error(f'Error loading {file_path}')
            raise

        return cls(pages=pages)


def get_pdf_page_count(file_path: str):
    try:
        with fitz.open(file_path) as doc:
            return doc.page_count
    except Exception as e:
        logger.error(f'get pdf page count error: {e}')
        raise


def get_pdf_metadata(file_path: str):
    try:
        with fitz.open(file_path) as doc:
            return doc.metadata
    except Exception as e:
        logger.error(f'get pdf metadata error: {e}')
        raise


from datetime import datetime, timedelta, timezone


def parse_pdf_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None

    # 去掉 D:
    if date_str.startswith('D:'):
        date_str = date_str[2:]

    # 正则拆解
    pattern = re.compile(
        r"""
        (?P<year>\d{4})
        (?P<month>\d{2})?
        (?P<day>\d{2})?
        (?P<hour>\d{2})?
        (?P<minute>\d{2})?
        (?P<second>\d{2})?
        (?:
            (?P<tz_sign>[+\-Z])
            (?P<tz_hour>\d{2})?'?
            (?P<tz_minute>\d{2})?'?
        )?
        """,
        re.VERBOSE
    )

    m = pattern.match(date_str)
    if not m:
        return None

    parts = m.groupdict()

    # 默认值
    year = int(parts['year'])
    month = int(parts['month'] or 1)
    day = int(parts['day'] or 1)
    hour = int(parts['hour'] or 0)
    minute = int(parts['minute'] or 0)
    second = int(parts['second'] or 0)

    dt = datetime(year, month, day, hour, minute, second)

    # time zone
    tz_sign = parts['tz_sign']
    if tz_sign:
        if tz_sign == 'Z':
            return dt.replace(tzinfo=timezone.utc)

        tz_hour = int(parts['tz_hour'] or 0)
        tz_minute = int(parts['tz_minute'] or 0)

        offset = timedelta(hours=tz_hour, minutes=tz_minute)

        if tz_sign == '-':
            offset = -offset

        return dt.replace(tzinfo=timezone(offset))

    return dt


if __name__ == '__main__':
    # file = '/Users/andersc/Downloads/cool nlp papers/Cognitive Architectures for Language Agents v3 (2024).pdf'
    # file = '/Users/andersc/data/papers/arxiv/2511.21631 - Qwen3-VL Technical Report.pdf'
    # file = '/Users/andersc/Downloads/papers/Fundamentals of Building Autonomous LLM Agents (2025.10).pdf'
    # file = '/Users/andersc/data/dev/local_kb/Who Will Monetize Truth - A Thesis for the Future of the Information Business (2026.03).pdf'
    file = '/Users/andersc/data/dev/local_kb/ThoughtWorks - Technology Radar 1269.pdf'
    # output_dir = '/Users/andersc/data/papers/pdf/LLM Agents'
    output_dir = None
    n_pages = 10

    print('metadata:', get_pdf_metadata(file))

    doc = PdfDocument.load_file(file, image_dir=output_dir, n_pages=n_pages)
    print(f'page count: {len(doc.pages)}\n')
    for page in doc.pages:
        if page.page_number > n_pages:
            break

        print(f'page {page.page_number}: ({page.width}, {page.height}), {len(page.blocks)} blocks:')

        for block in page.blocks:
            # print(f'block: {block.type}, {block.page_number}, {block.block_number}')
            if block.is_text():
                print(block.text[:200])
                print('\n')

                # if 'Background of LLMs' in block.text:
                #     print('block sample:')
                #     for line in block.lines:
                #         for span in line.spans:
                #             print(span)
                #             print()

            else:
                if block.image_path:
                    print('image:', block.image_path)
                    print(block.bbox)
                    print()

        print('\n')
