# coding=utf-8
import logging
import re
from collections import Counter
from pathlib import Path
from datetime import datetime, timedelta, timezone

import fitz
from pydantic import BaseModel, PrivateAttr, Field

from archaeo.iterable import rename_keys
from archaeo.maths import Rectangle
from archaeo.texts import contains_zh_or_alphabet

logger = logging.getLogger(__name__)

FLAG_SUPERSCRIPT = 1 << 0
FLAG_ITALIC = 1 << 1
FLAG_SERIFED = 1 << 2
FLAG_MONOSPACED = 1 << 3
FLAG_BOLD = 1 << 4

_CAPTION_PATTERNS = [
    r'^(figure|fig\.?)\s+\d+\b',
    r'^table\s+\d+\b',
    # r'^algorithm\s+\d+\b',
    # r'^listing\s+\d+\b',
    # r'^code\s+listing\s+\d+\b',
]


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

    @property
    def spans(self) -> list[PdfSpan]:
        return sum([line.spans for line in self.lines], [])

    @property
    def first_span(self) -> PdfSpan:
        return self.lines[0].spans[0]

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
                  n_pages: int | None = None):

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


class PdfDocSection(BaseModel):
    title: str | None = None
    level: int = 1
    title_block: PdfBlock | None = None
    blocks: list[PdfBlock] = Field(default_factory=list)
    children: list['PdfDocSection'] = Field(default_factory=list)


class PdfDocSections(BaseModel):
    source: str | None = None
    sections: list[PdfDocSection] = Field(default_factory=list)


def normalize_header_footer_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '%d', text)
    return text


def clean_pdf_header_footer(doc: PdfDocument) -> PdfDocument:
    if len(doc.pages) < 2:
        return doc

    page_width, page_height = doc.pages[0].width, doc.pages[0].height
    # print(f'page: ({page_width}, {page_height})')

    header_footer_threshold = 0.1
    header_limit = page_height * header_footer_threshold
    footer_limit = page_height * (1-header_footer_threshold)

    n_pages = len(doc.pages)
    threshold = max(2, round(n_pages * 0.6))
    # print(f'threshold: {threshold}')

    candidate_blocks = []
    direct_remove = set()

    for page in doc.pages:
        for block in page.blocks:
            if not block.is_text():
                continue

            if len(block.text) > 200:
                continue

            in_header = block.bbox.bottom <= header_limit
            in_footer = block.bbox.top >= footer_limit

            if not (in_header or in_footer):
                continue

            zone = 'header' if in_header else 'footer'
            norm_text = normalize_header_footer_text(block.text)
            if norm_text == '%d':
                direct_remove.add((block.page_number, block.block_number))
                continue
            candidate_blocks.append((block, zone, norm_text))

    c_texts = Counter((zone, nbt) for _, zone, nbt in candidate_blocks)
    target_patterns = {k for k, cnt in c_texts.most_common() if cnt >= threshold}
    # print(c_texts.most_common())
    print('target_patterns:', target_patterns)
    # target_blocks = [b for b, t in candidate_blocks if t in target_patterns]
    # print('target_blocks:', len(target_blocks))
    # for tb in target_blocks:
    #     print(tb.page_number, tb.block_number, tb.text)
    repeated_remove = {(block.page_number, block.block_number)
                       for block, zone, norm_text in candidate_blocks
                       if (zone, norm_text) in target_patterns}
    block_info_to_remove = direct_remove | repeated_remove
    print('to_remove:', sorted(block_info_to_remove))

    new_pages = []
    for page in doc.pages:
        new_blocks = [b for b in page.blocks if (b.page_number, b.block_number) not in block_info_to_remove]
        # if new_blocks:
        page.blocks = new_blocks
        new_pages.append(page)

    return PdfDocument(pages=new_pages)


def is_caption_text(text: str) -> bool:
    text = text.strip()

    return any(
        re.match(pattern, text, re.I)
        for pattern in _CAPTION_PATTERNS
    )


# def is_list_item_text(text: str) -> bool:
#     return False


def build_document_sections(doc: PdfDocument,
                            *,
                            title_threshold: float=3.9) -> PdfDocSections:

    def estimate_body_font(_doc: PdfDocument) -> tuple:
        threshold = 0.8

        c_font_sizes = Counter()
        c_font_names = Counter()
        c_font_colors = Counter()
        for page in _doc.pages:
            for block in page.blocks:
                if not block.is_text():
                    continue
                for line in block.lines:
                    for span in line.spans:
                        # print(round(span.font_size, 1), span.font_name, span.font_color, span.is_bold())
                        span_text_count = len(span.text)
                        c_font_sizes[round(span.font_size, 1)] += span_text_count
                        c_font_names[span.font_name] += span_text_count
                        c_font_colors[span.font_color] += span_text_count

        print(c_font_sizes.most_common())
        print(c_font_names.most_common())
        print(c_font_colors.most_common())

        if not c_font_sizes:
            return None, None, None

        most_common_font_size, most_common_fs_count = c_font_sizes.most_common(1)[0]
        most_common_font_name, fn_count = c_font_names.most_common(1)[0]
        most_common_font_color, fc_count = c_font_colors.most_common(1)[0]
        print(most_common_font_size, most_common_fs_count / c_font_sizes.total())
        print(most_common_font_name, fn_count / c_font_names.total())
        print(most_common_font_color, fc_count / c_font_colors.total())

        font_size = None
        if most_common_fs_count / c_font_sizes.total() > threshold:
            font_size = most_common_font_size
        else:
            accum_fs_cnt = 0
            for fs_item, fs_item_cnt in c_font_sizes.most_common():
                if abs(most_common_font_size - fs_item) <= 0.2:
                    accum_fs_cnt += fs_item_cnt
                else:
                    break
            if accum_fs_cnt / c_font_sizes.total() > threshold:
                font_size = most_common_font_size
                print(most_common_font_size, accum_fs_cnt / c_font_sizes.total())

        font_name = None
        if fn_count / c_font_names.total() > threshold:
            font_name = most_common_font_name
        font_color = None
        if fc_count / c_font_colors.total() > threshold:
            font_color = most_common_font_color
        print(f'detected font: {font_size}, {font_name}, {font_color}')
        return font_size, font_name, font_color

    def is_vertical_block(_block: PdfBlock) -> bool:
        return _block.bbox.height >= (_block.bbox.width * 2.0)

    def score_section_title(
            _block,
            body_font_size: float,
            prev_block=None,
            next_block=None,
    ) -> float:
        score = 0.0

        if body_font_size is None:
            return 0.0

        text = _block.text.strip()
        if not text:
            return 0.0

        # short text
        if len(text) <= 80:
            score += 1.0
        elif len(text) <= 160:
            score += 0.5
        else:
            score -= 2.0

        # line count
        if len(_block.lines) == 1:
            score += 1.0
        elif len(_block.lines) <= 3:
            score += 0.5
        else:
            score -= 2.0

        # font size
        cur_font_size = _block.first_span.font_size
        size_diff = cur_font_size - body_font_size
        if size_diff >= 4:
            score += 3.0
        elif size_diff >= 2:
            score += 2.0
        elif size_diff >= 1:
            score += 1.0

        if cur_font_size < body_font_size:
            if cur_font_size < body_font_size - 3:
                score -= 5.0
            elif cur_font_size < body_font_size - 2:
                score -= 2.0
            elif cur_font_size < body_font_size - 1:
                score -= 1.0
            elif cur_font_size < body_font_size - 0.5:
                score -= 0.5

        # bold
        if _block.first_span.is_bold():
            score += 1.5

        # vertical spacing
        if prev_block is not None:
            gap_before = _block.bbox.top - prev_block.bbox.bottom
            if gap_before > body_font_size * 2.0:
                score += 2.0
            elif gap_before > body_font_size * 1.0:
                score += 1.0

        if next_block is not None:
            gap_after = next_block.bbox.top - _block.bbox.bottom
            if gap_after > body_font_size * 2.0:
                score += 1.0
            elif gap_after > body_font_size * 1.0:
                score += 0.5

        # # bad signs
        if is_caption_text(text):
            score -= 3.0

        # if is_list_item_text(text):
        #     score -= 2.0

        if text.endswith('.'):
            score -= 0.8

        if not contains_zh_or_alphabet(_block.text):
            score -= 5.0

        if is_vertical_block(_block):
            score -= 5.0

        return score

    body_font_size, body_font_name, body_font_color = estimate_body_font(doc)

    sections = []
    current = PdfDocSection(title=None, level=0, blocks=[])

    # if body_font_size is None:
    #     return doc

    for page in doc.pages:
        blocks = page.blocks

        for i, block in enumerate(blocks):
            if not block.is_text():
                current.blocks.append(block)
                continue

            prev_block = blocks[i-1] if i > 0 else None
            next_block = blocks[i+1] if i < len(blocks) - 1 else None
            score = score_section_title(block,
                                        body_font_size=body_font_size,
                                        prev_block=prev_block,
                                        next_block=next_block)

            if score >= title_threshold:

                print(f'section title found: score: {score}')
                print('page:', block.page_number, 'text:', block.text)
                print()

                if current.title is not None or current.blocks:
                    sections.append(current)

                current = PdfDocSection(title=block.text.strip(),
                                        level=1,
                                        title_block=block,
                                        blocks=[])
            else:
                current.blocks.append(block)

    if current.title is not None or current.blocks:
        sections.append(current)

    return PdfDocSections(sections=sections)


def clean_pdf_doc(doc: PdfDocument) -> PdfDocument:
    doc = clean_pdf_header_footer(doc)
    return doc


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
    # file = '/Users/andersc/data/dev/local_kb/ThoughtWorks - Technology Radar 1269.pdf'
    # file = '/Users/andersc/data/dev/local_kb/Stanford_ai_index_report_2026.pdf'
    file = '/Users/andersc/data/dev/local_kb/TheEconomist.2026.05.09.pdf'
    # file = '/Users/andersc/data/dev/local_kb/DeepSeek-V4 - Towards Highly Efficient Million-Token Context Intelligence (2026.04).pdf'
    # file = '/Users/andersc/Downloads/八分半/看理想十年之选长名单（人生书单内部资料）.pdf'
    # output_dir = '/Users/andersc/data/papers/pdf/LLM Agents'
    output_dir = None
    n_pages = 50

    print('metadata:', get_pdf_metadata(file))

    doc = PdfDocument.load_file(file, image_dir=output_dir, n_pages=n_pages)
    print(f'page count: {len(doc.pages)}\n')
    clean_pdf_doc(doc)
    doc_sections = build_document_sections(doc)
    for sec in doc_sections.sections:
        print(sec.title, sec.level, len(sec.blocks))
        print()

    # for page in doc.pages:
    #     if page.page_number > n_pages:
    #         break
    #
    #     print(f'page {page.page_number}: ({page.width}, {page.height}), {len(page.blocks)} blocks:')
    #
    #     for block in page.blocks:
    #         # print(f'block: {block.type}, {block.page_number}, {block.block_number}')
    #         print(block.page_number, block.block_number, block.bbox)
    #         if block.is_text():
    #             print(block.text[:20000])
    #             print()
    #
    #             # if 'Background of LLMs' in block.text:
    #             #     print('block sample:')
    #             #     for line in block.lines:
    #             #         for span in line.spans:
    #             #             print(span)
    #             #             print()
    #
    #         else:
    #             if block.image_path:
    #                 print('image:', block.image_path)
    #                 print(block.bbox)
    #                 print()
    #
    #     print('\n')
