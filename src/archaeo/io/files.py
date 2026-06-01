# coding=utf-8
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Iterator


def list_files(
        directory: str | Path,
        pattern: str = "*",
        excludes: Optional[Callable[[Path], bool]] = None
) -> Iterator[Path]:
    """
    Iterate a directory recursively, yield the absolute file paths.

    Args:
        directory: target dir.
        pattern: glob pattern for filenames (e.g. '*.pdf')
        excludes: function to exclude files

    Yields:
        Absolute Path objects.

    Example:
        # exclude all files under .git dir.
        list_files(".", excludes=lambda p: ".git" in p.parts)
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    if excludes is None:
        excludes = lambda _: False

    for file_path in directory.rglob(pattern):
        if file_path.is_file() and not excludes(file_path):
            yield file_path


def iter_files(dirs, pattern='*.*', excludes=None):
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    for d in dirs:
        for fp in list_files(d, pattern, excludes):
            yield fp


def get_file_created_time(path: str | Path) -> datetime | None:
    stat = Path(path).stat()

    birthtime = getattr(stat, 'st_birthtime', None)
    if birthtime is None:
        return None

    return datetime.fromtimestamp(birthtime)


def get_file_modified_time(path: str | Path) -> datetime:
    return datetime.fromtimestamp(
        Path(path).stat().st_mtime
    )


def get_file_accessed_time(path: str | Path) -> datetime:
    return datetime.fromtimestamp(
        Path(path).stat().st_atime
    )


def get_file_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def get_file_extension(file_path: str | Path):
    return Path(file_path).suffix


def get_file_name(file_path: str | Path, with_ext: bool=True):
    if with_ext:
        return Path(file_path).name
    return Path(file_path).stem


def touch_file(file_path: str | Path):
    p = Path(file_path)
    p.touch()


def json_load(file: str | Path):
    file = Path(file)
    with file.open('r', encoding='utf-8') as fin:
        return json.load(fin)


def json_dump(obj, file: str | Path, ensure_ascii=False, indent=2, create_parent: bool=True):
    file = Path(file)
    if create_parent:
        file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w', encoding='utf-8') as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent)
