# coding=utf-8
import subprocess
from pathlib import Path
from typing import Iterable


class ContentTypes:
    pdf = 'com.adobe.pdf'
    epub = 'org.idpf.epub-container'
    markdown = 'net.daringfireball.markdown'

    text = 'public.plain-text'
    jpeg = 'public.jpeg'
    png = 'public.png'


content_types = {k: v for k, v in ContentTypes.__dict__.items() if not k.startswith('__')}
# print(content_types)


def mdfind(
    query: str,
    *,
    only_in: str | Path | None = None,
    search_on_names: bool = False,
    max_results: int | None = None,
    timeout: float = 10.0,
) -> list[Path]:
    """
    Run macOS mdfind and return matched file paths.

    Parameters
    ----------
    query:
        Spotlight query string.
        Examples:
            "python"
            'kMDItemFSName == "*.pdf"'
            'kMDItemTextContent == "*机器学习*"'

    only_in:
        Restrict search to a directory, equivalent to `mdfind -onlyin DIR`.

    search_on_names:
        Use `mdfind -name QUERY`, useful for simple filename search.

    max_results:
        Limit number of returned paths in Python after mdfind returns.

    timeout:
        subprocess timeout in seconds.
    """
    cmd = ["mdfind"]

    if only_in is not None:
        cmd.extend(["-onlyin", str(Path(only_in).expanduser().resolve())])

    if search_on_names:
        cmd.extend(["-name", query])
    else:
        cmd.append(query)

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"mdfind timed out after {timeout} seconds") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"mdfind failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    paths = [Path(line) for line in result.stdout.splitlines() if line.strip()]

    if max_results is not None:
        paths = paths[:max_results]

    return paths


def search_by_name(file_name: str):
    return mdfind(file_name, search_on_names=True)


def search_files(name: str, extension: str | None = None):
    results = mdfind(name, search_on_names=True)
    if extension:
        suffix = '.' + extension.lstrip('.').lower()
        results = [fp for fp in results if Path(fp).suffix.lower() == suffix]
    return results


if __name__ == '__main__':
    # print(mdfind('中国古典', search_on_names=True))
    # print(mdfind('中国古典', search_on_names=True, only_in='/Volumes/T2/books/kindle'))
    print()
    for file in search_files('極致愛撫', extension=''):
        print(file)
