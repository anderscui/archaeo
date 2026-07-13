# coding=utf-8
from pathlib import Path

from archaeo import logger
from archaeo.io.audio import get_audio_metadata
from archaeo.io.docs import LocalFileMetadata
from archaeo.io.epub import get_epub_metadata
from archaeo.io.files import get_absolute_path
from archaeo.io.image import get_image_metadata
from archaeo.io.markdown import get_markdown_metadata
from archaeo.io.office import get_docx_metadata
from archaeo.io.pdf import get_pdf_metadata
from archaeo.io.video import get_video_metadata

AUDIO_EXTENSIONS = {
    "mp3", "wav", "aiff", "aif", "flac", "m4a", "aac", "ogg",
}

VIDEO_EXTENSIONS = {
    "mp4", "m4v", "mov", "mkv", "avi", "wmv", "flv", "mpg", "mpeg",
}

IMAGE_EXTENSIONS = {
    "png", "jpg", "jpeg", "gif", "webp", "bmp", "heic", "heif",
}


def get_local_file_metadata(file_path: str | Path) -> LocalFileMetadata:
    path = get_absolute_path(file_path)
    ext = path.suffix.lower().lstrip(".")

    if not ext:
        return LocalFileMetadata()

    try:
        if ext == "pdf":
            return get_pdf_metadata(path)

        if ext == "epub":
            return get_epub_metadata(path)

        if ext == "docx":
            return get_docx_metadata(path)

        if ext == "md":
            return get_markdown_metadata(path)

        if ext in IMAGE_EXTENSIONS:
            return get_image_metadata(path)

        if ext in AUDIO_EXTENSIONS:
            return get_audio_metadata(path)

        if ext in VIDEO_EXTENSIONS:
            return get_video_metadata(path)

        return LocalFileMetadata()

    except Exception as e:
        logger.warning(f"get local file metadata failed: {e}, file={path}")
        return LocalFileMetadata()


if __name__ == '__main__':
    # file = '~/Downloads/movie/恶魔奶爸推介的英语听力口语大杀器《EnglishPod》/EnglishPod 1-50/0001/englishpod_B0001pr.mp3'
    # print(get_local_file_metadata(file))
    #
    # file = '~/Downloads/mac setup.docx'
    # meta = get_local_file_metadata(file)
    # for item in meta.outline.items:
    #     print(item.title, item.level)
    # print()
    # print(meta.metadata)
    # print(meta.preview)

    file = '~/Downloads/豆瓣年度 2025/So Late in the Day (Claire Keegan, 2023).epub'
    print(get_epub_metadata(file))
    # get_epub_toc('~/Downloads/豆瓣年度 2025/Nine Stories (J. D. Salinger, 2019).epub')
    # get_epub_toc('~/Downloads/理想国系列书单/6世界运行的逻辑：理想國探索世界入门书单合集（全14册）/世界运行的逻辑：理想國探索世界入门书单合集（全14册）.epub')
    # get_epub_toc('~/Downloads/豆瓣年度 2025/身分政治：民粹崛起、民主倒退，認同與尊嚴的鬥爭為何席捲當代世界？= Identity The Demand for Dignity and the Politics of Resentment (法蘭西斯 · 福山, 2020).mobi')
    # get_epub_toc('~/Downloads/豆瓣年度 2025/Tales from the Cafe (Before the Coffee Gets Cold 2) (Toshikazu Kawaguchi, 2021).mobi')
