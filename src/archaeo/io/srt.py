# coding=utf-8
import re
from dataclasses import dataclass
from pathlib import Path

from archaeo.io.files import get_absolute_path, write_text

TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})"
    r"\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)

SENTENCE_END_RE = re.compile(r'[.!?]["\']?$')


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


def parse_timestamp(value: str) -> float:
    """Convert SRT timestamp to seconds."""
    hours, minutes, seconds = value.split(":")
    seconds, millis = seconds.split(",")

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000
    )


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp."""
    millis = round(seconds * 1000)

    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def format_short_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    seconds = round(seconds)

    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def parse_srt(path: str | Path) -> list[SubtitleCue]:
    """Parse a basic SRT file."""
    path = Path(path).expanduser()
    text = path.read_text(encoding="utf-8")

    blocks = re.split(r"\n\s*\n", text.strip())

    cues = []

    for block in blocks:
        lines = block.splitlines()

        for i, line in enumerate(lines):
            match = TIMESTAMP_RE.search(line)
            if not match:
                continue

            subtitle_text = " ".join(
                x.strip()
                for x in lines[i + 1:]
                if x.strip()
            )

            subtitle_text = re.sub(
                r"\s+",
                " ",
                subtitle_text,
            ).strip()

            if subtitle_text:
                cues.append(
                    SubtitleCue(
                        start=parse_timestamp(match.group("start")),
                        end=parse_timestamp(match.group("end")),
                        text=subtitle_text,
                    )
                )

            break

    return cues


def split_sentences(text: str) -> list[str]:
    """Simple English sentence splitter."""
    return [
        x.strip()
        for x in re.split(r"(?<=[.!?])\s+", text)
        if x.strip()
    ]


def srt_to_text(
    path: str | Path,
    *,
    max_paragraph_chars: int = 800,
    max_paragraph_sentences: int = 6,
) -> str:
    """
    Convert SRT to plain transcript text.

    The result contains no timestamps and groups sentences
    into reasonably sized paragraphs.
    """
    cues = parse_srt(path)

    text = " ".join(cue.text for cue in cues)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = split_sentences(text)

    paragraphs = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if current and (
            len(candidate) > max_paragraph_chars
            or len(current) >= max_paragraph_sentences
        ):
            paragraphs.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        paragraphs.append(" ".join(current))

    return "\n\n".join(paragraphs)


def srt_to_timed_text(
    path: str | Path,
    *,
    min_seconds: float = 20,
    max_seconds: float = 30,
) -> str:
    """
    Convert SRT to coarse timed transcript.

    Each paragraph is roughly min_seconds to max_seconds long.
    After min_seconds, prefer splitting at sentence endings.
    """
    cues = parse_srt(path)

    groups = []
    current = []

    for cue in cues:
        current.append(cue)

        duration = current[-1].end - current[0].start

        sentence_end = bool(
            SENTENCE_END_RE.search(cue.text)
        )

        should_split = (
            duration >= max_seconds
            or (
                duration >= min_seconds
                and sentence_end
            )
        )

        if should_split:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    output = []

    for group in groups:
        start = group[0].start
        end = group[-1].end

        text = " ".join(cue.text for cue in group)
        text = re.sub(r"\s+", " ", text).strip()

        output.append(
            f"[{format_short_timestamp(start)} - "
            f"{format_short_timestamp(end)}]\n"
            f"{text}"
        )

    return "\n\n".join(output)


def resegment_srt(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_chars: int = 100,
) -> None:
    """
    Merge fragmented SRT cues into more readable subtitle blocks.

    Prefer splitting at sentence endings.
    If a block becomes too long, split at the current cue boundary.

    Original cues are never split, so timestamps remain approximate
    but are not artificially invented.
    """
    cues = parse_srt(input_path)

    groups = []
    current = []

    for cue in cues:
        current.append(cue)

        text = " ".join(x.text for x in current)

        sentence_end = bool(
            SENTENCE_END_RE.search(cue.text)
        )

        too_long = len(text) >= max_chars

        if sentence_end or too_long:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    blocks = []

    for i, group in enumerate(groups, start=1):
        start = group[0].start
        end = group[-1].end

        text = " ".join(cue.text for cue in group)
        text = re.sub(r"\s+", " ", text).strip()

        blocks.append(
            f"{i}\n"
            f"{format_timestamp(start)} --> "
            f"{format_timestamp(end)}\n"
            f"{text}"
        )

    output_path = Path(output_path).expanduser()
    output_path.write_text(
        "\n\n".join(blocks) + "\n",
        encoding="utf-8",
    )


if __name__ == '__main__':
    srt_file = '~/Downloads/AndrewNgOnAI.en.srt'
    srt_text_file = '~/Downloads/AndrewNgOnAI.en.txt'
    write_text(srt_text_file, srt_to_text(srt_file))
