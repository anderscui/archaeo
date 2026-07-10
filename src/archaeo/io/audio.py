# coding=utf-8
from pathlib import Path
from typing import Any

from mutagen import File as MutagenFile
from archaeo.io.docs import LocalFileMetadata


def get_audio_metadata(file_path: str | Path) -> LocalFileMetadata:
    audio = MutagenFile(str(file_path), easy=True)
    metadata: dict[str, Any] = {}

    if audio is None:
        return LocalFileMetadata(metadata=metadata)

    info = getattr(audio, "info", None)
    if info:
        metadata["duration"] = getattr(info, "length", None)
        metadata["bitrate"] = getattr(info, "bitrate", None)
        metadata["sample_rate"] = getattr(info, "sample_rate", None)

    for key in ("title", "artist", "album", "albumartist", "date", "genre", "tracknumber"):
        val = audio.get(key)
        if val:
            metadata[key] = val

    return LocalFileMetadata(metadata=metadata)
