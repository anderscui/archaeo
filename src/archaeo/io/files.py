# coding=utf-8
import hashlib
import json
import os
import shutil
import zipfile

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Iterator

from archaeo import logger


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
    directory = get_absolute_path(directory)
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


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()


def is_dir(path: str | Path) -> bool:
    return Path(path).is_dir()


def is_file(path: str | Path) -> bool:
    return Path(path).is_file()


def is_absolute_path(path: str | Path) -> bool:
    return Path(path).is_absolute()


def expand_user_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def parent_of_path(path: str | Path) -> Path:
    return Path(path).parent


def get_absolute_path(raw_path: str | Path) -> Path:
    return Path(raw_path).expanduser().resolve()


def get_file_created_time(path: str | Path, use_timestamp: bool=True) -> int | datetime | None:
    stat = Path(path).stat()

    birthtime = getattr(stat, 'st_birthtime', None)
    if birthtime is None:
        return None

    return int(birthtime * 1000) if use_timestamp else datetime.fromtimestamp(birthtime)


def get_file_modified_time(path: str | Path, use_timestamp: bool=True) -> int | datetime:
    st = Path(path).stat().st_mtime
    return int(st * 1000) if use_timestamp else datetime.fromtimestamp(st)


def get_file_accessed_time(path: str | Path, use_timestamp: bool=True) -> int | datetime:
    st = Path(path).stat().st_atime
    return int(st * 1000) if use_timestamp else datetime.fromtimestamp(st)


def get_file_size(path: str | Path) -> int:
    return Path(path).stat().st_size


def get_file_extension(file_path: str | Path):
    return Path(file_path).suffix


def get_file_name(file_path: str | Path, with_ext: bool=True):
    if with_ext:
        return Path(file_path).name
    return Path(file_path).stem


def get_file_blocks(path: str | Path) -> int:
    path = expand_user_path(path)
    return path.stat(follow_symlinks=False).st_blocks


def is_relative_to_any(path: str | Path, paths: list[Path]):
    path = get_absolute_path(path)
    return any(path.is_relative_to(op) for op in paths)


def get_file_hash(
    path: str | Path,
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> str:
    path = expand_user_path(path)

    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def read_lines(path: str | Path, encoding: str = "utf-8"):
    path = expand_user_path(path)
    with path.open("r", encoding=encoding) as f:
        yield from f


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    path = expand_user_path(path)
    return path.read_text(encoding=encoding)


def touch_file(file_path: str | Path):
    p = Path(file_path)
    p.touch()


def ensure_parent_dir(path: str | Path) -> Path:
    path = expand_user_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def remove_file(path: str | Path) -> None:
    path = Path(path)

    try:
        path.unlink()
    except OSError:
        logger.error(f'Failed to delete file: {path}')


def remove_dir(path: str | Path) -> None:
    path = Path(path)

    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError:
        logger.error(f'Failed to delete directory: {path}')


def remove_files(paths: Iterable[str | Path]) -> None:
    for path in paths:
        remove_file(path)


def is_executable(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and os.access(path, os.X_OK)


def write_lines(
    path: str | Path,
    lines: Iterable[str],
    encoding: str = "utf-8",
) -> None:
    path = expand_user_path(path)
    ensure_parent_dir(path)
    path.write_text('\n'.join(lines), encoding=encoding)


def write_text(
    path: str | Path,
    text: str,
    encoding: str = "utf-8",
) -> None:
    path = expand_user_path(path)
    ensure_parent_dir(path)
    path.write_text(text, encoding=encoding)


def rename_file(src: str | Path, dst: str | Path) -> None:
    src = expand_user_path(src)
    dst = expand_user_path(dst)
    ensure_parent_dir(dst)
    src.rename(dst)


def copy_file(
    src: str | Path,
    dst: str | Path,
    overwriting: bool = True,
) -> None:
    src = expand_user_path(src)
    dst = expand_user_path(dst)
    ensure_parent_dir(dst)

    if dst.exists() and not overwriting:
        logger.warning(f'File exists: {dst}')
        return

    shutil.copy2(src, dst)


def move_file(
    src: str | Path,
    dst: str | Path,
    overwriting: bool = True,
) -> None:
    src = expand_user_path(src)
    dst = expand_user_path(dst)
    ensure_parent_dir(dst)

    if dst.exists() and not overwriting:
        logger.warning(f'File exists: {dst}')
        return

    shutil.move(src, dst)


def json_load(file: str | Path):
    file = get_absolute_path(file)
    with file.open('r', encoding='utf-8') as fin:
        return json.load(fin)


def json_dump(obj, file: str | Path, ensure_ascii=False, indent=2, create_parent: bool=True):
    file = get_absolute_path(file)
    if create_parent:
        file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('w', encoding='utf-8') as fout:
        json.dump(obj, fout, ensure_ascii=ensure_ascii, indent=indent)


def unzip_file(
    zip_file: str | Path,
    target_dir: str | Path,
) -> Path:
    zip_file = expand_user_path(zip_file)
    target_dir = expand_user_path(target_dir)

    output_dir = target_dir / zip_file.stem

    if output_dir.exists():
        logger.warning(f'Unzip target exists: {output_dir}')

    with zipfile.ZipFile(zip_file) as archive:
        archive.extractall(output_dir)

    return output_dir


def show_zipfile_contents(zip_file: str | Path) -> None:
    zip_file = expand_user_path(zip_file)
    with zipfile.ZipFile(zip_file) as archive:
        archive.printdir()


def zip_dir(
    source_dir: str | Path,
    output_file: str | Path | None = None,
) -> Path:
    source_dir = expand_user_path(source_dir)
    if not source_dir.is_dir():
        raise ValueError(f'Not a directory: {source_dir}')

    if output_file is None:
        output_file = source_dir.with_suffix('.zip')
    else:
        output_file = expand_user_path(output_file)

    with zipfile.ZipFile(output_file, 'w') as archive:
        for file in list_files(source_dir):
            file = Path(file)

            if file.is_dir():
                continue

            archive.write(
                file,
                arcname=file.relative_to(source_dir),
            )

    return output_file


def compare_dirs(dir1: str| Path, dir2: str | Path) -> bool:
    dir1 = get_absolute_path(dir1)
    dir2 = get_absolute_path(dir2)
    files1 = list(list_files(dir1))
    files2 = list(list_files(dir2))
    if len(files1) != len(files2):
        return False

    hashes1 = sorted([get_file_hash(f) for f in files1])
    hashes2 = sorted([get_file_hash(f) for f in files2])
    return hashes1 == hashes2


if __name__ == '__main__':
    assert is_relative_to_any('~/Downloads/', [Path('/Users/andersc/Downloads')])
