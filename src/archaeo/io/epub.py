# coding=utf-8
from pathlib import Path

import fitz

from archaeo import logger
from archaeo.io.docs import Toc, LocalFileMetadata
from archaeo.io.files import get_absolute_path

fitz.TOOLS.mupdf_display_warnings(False)
fitz.TOOLS.mupdf_display_errors(False)


def get_epub_metadata(file_path: str | Path,
                     max_outline_level: int = 2) -> LocalFileMetadata:
    file_path = get_absolute_path(file_path)
    try:
        with fitz.open(file_path) as doc:
            metadata = dict(doc.metadata or {})
            metadata['page_count'] = doc.page_count

            outlines = []
            toc = doc.get_toc(simple=False)
            for item in toc:
                # page is 1-based
                lvl, title, page = item[:3]
                if title.strip() and lvl <= max_outline_level:
                    outlines.append(item)

        return LocalFileMetadata(metadata=metadata, outline=Toc.load(outlines))
    except Exception as e:
        logger.error(f'get epub metadata error: {e}')
        raise


if __name__ == '__main__':
    # zipfile.BadZipFile: File is not a zip file
    # file = '~/Downloads/to_read/理想国系列书单/1理想国译丛系列（61册）/060创造欧洲人：现代性的诞生与欧洲文化的形塑/创造欧洲人：现代性的诞生与欧洲文化的形塑1.epub'
    file = '~/Downloads/to_read/理想国系列书单/1理想国译丛系列（61册）/053与屠刀为邻：幸存者、刽子手与卢旺达大tu杀的记忆/__【理想国译丛053】与屠刀为邻：幸存者、刽子手与卢旺达大tu杀的记忆.epub'
    print(get_epub_metadata(file))
