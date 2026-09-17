# coding=utf-8
from pathlib import Path

from archaeo import logger
from archaeo.io.files import get_absolute_path, IMAGE_EXTENSIONS, read_text
from archaeo.io.ocr.ocr import ocr_by_llm_openai
from archaeo.llm_providers import BaseLlmProvider, OpenRouterModels, OpenRouterProvider


def get_file_text(file_path: str | Path,
                  ocr_provider: BaseLlmProvider) -> str | None:
    path = get_absolute_path(file_path)
    ext = path.suffix.lower().lstrip(".")

    if not ext:
        return None

    try:
        # if ext == "pdf":
        #     return get_pdf_metadata(path)
        #
        # if ext == "epub":
        #     return get_epub_metadata(path)
        #
        # if ext == "docx":
        #     return get_docx_metadata(path)

        if ext in IMAGE_EXTENSIONS:
            result = ocr_by_llm_openai(ocr_provider, path)
            if not result:
                return None
            return result.get('text') or None
        elif ext in {'txt'}:
            return read_text(path)
        elif ext in {'md'}:
            return read_text(path)

        # if ext in AUDIO_EXTENSIONS:
        #     return get_audio_metadata(path)
        #
        # if ext in VIDEO_EXTENSIONS:
        #     return get_video_metadata(path)

        return None

    except Exception as e:
        logger.warning(f"get local file metadata failed: {e}, file={path}")
        return None


if __name__ == '__main__':
    llm_ocr = OpenRouterProvider(OpenRouterModels.gemini_flash_lite_3_1)
    # file = '~/Downloads/jobs/py-202608-renamed/jd3.png'
    # print(file)
    # print(get_file_text(file, ocr_provider=llm_ocr))

    file = '/Users/andersc/Downloads/jobs/py-202609-renamed/jd1.md'
    print(file)
    print(get_file_text(file, ocr_provider=llm_ocr))
