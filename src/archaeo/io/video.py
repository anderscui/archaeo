# coding=utf-8
import json
import shutil
import subprocess

from pathlib import Path
from typing import Any

from archaeo import logger
from archaeo.io.docs import LocalFileMetadata, Toc, TocItem
from archaeo.io.files import get_absolute_path


def get_video_metadata(file_path: str | Path, timeout: int = 10) -> LocalFileMetadata:
    if shutil.which('ffprobe') is None:
        logger.warning(f'ffprobe not found, skip video metadata: {file_path}')
        return LocalFileMetadata()

    file_path = get_absolute_path(file_path)

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        str(file_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )

        if proc.returncode != 0:
            logger.warning(
                "ffprobe failed: returncode=%s, stderr=%s, file=%s",
                proc.returncode,
                proc.stderr.strip(),
                file_path,
            )
            return LocalFileMetadata()

        data = json.loads(proc.stdout or "{}")

        metadata: dict[str, Any] = {}

        fmt = data.get("format") or {}
        tags = fmt.get("tags") or {}

        if fmt.get("format_name"):
            metadata["format_name"] = fmt["format_name"]

        if fmt.get("duration"):
            metadata["duration"] = float(fmt["duration"])

        if fmt.get("size"):
            metadata["size"] = int(fmt["size"])

        if fmt.get("bit_rate"):
            metadata["bit_rate"] = int(fmt["bit_rate"])

        for key in ("title", "artist", "album", "album_artist", "date", "year", "genre", "comment"):
            if tags.get(key):
                metadata[key] = tags[key]

        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue

            metadata["video_codec"] = stream.get("codec_name")
            metadata["width"] = stream.get("width")
            metadata["height"] = stream.get("height")
            metadata["fps"] = stream.get("avg_frame_rate")

            stream_tags = stream.get("tags") or {}
            if stream_tags.get("rotate"):
                metadata["rotate"] = stream_tags["rotate"]

            break

        audio_streams = [
            s for s in data.get("streams", [])
            if s.get("codec_type") == "audio"
        ]
        if audio_streams:
            metadata["audio_count"] = len(audio_streams)
            metadata["audio_codec"] = audio_streams[0].get("codec_name")

        chapters = data.get("chapters") or []
        outline_items = []

        for idx, chapter in enumerate(chapters, start=1):
            chapter_tags = chapter.get("tags") or {}
            title = chapter_tags.get("title") or f"Chapter {idx}"

            outline_items.append(
                TocItem(
                    level=1,
                    title=title,
                    page_number=None,
                )
            )

        return LocalFileMetadata(
            metadata=metadata,
            outline=Toc(items=outline_items),
        )

    except subprocess.TimeoutExpired:
        logger.warning("ffprobe timeout: file=%s", file_path)
        return LocalFileMetadata()

    except Exception as e:
        logger.warning("get video metadata failed: %s, file=%s", e, file_path)
        return LocalFileMetadata()


if __name__ == '__main__':
    file = '/Volumes/T2/media/youtube/八分半 YT.mp4'
    # file = '~/Downloads/t2-media/The Interview-Want to ‘Optimize’ Your Happiness-1080p.mp4'
    # file = '~/Downloads/yt/十三邀/【十三邀 第一季】第7期：许知远对话张楚 [9xOI2pjsPS8].mp4'
    print(get_video_metadata(file))
    print(get_video_metadata2(file))
