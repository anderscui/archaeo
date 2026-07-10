# coding=utf-8
from pathlib import Path

import fitz

from archaeo import logger
from archaeo.io.docs import Toc, LocalFileMetadata


def get_epub_metadata(file_path: str | Path,
                     max_outline_level: int = 2) -> LocalFileMetadata:
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
