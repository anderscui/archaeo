# coding=utf-8
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lz4.block
from pydantic import BaseModel, Field


MOZLZ4_MAGIC = b'mozLz40\0'


class FirefoxGroup(BaseModel):
    id: str
    name: str | None = None
    color: str | None = None
    collapsed: bool = False
    save_on_window_close: bool = True


class FirefoxTab(BaseModel):
    index: int
    title: str | None = None
    url: str

    last_accessed: datetime | None = None
    favicon: str | None = None

    window_index: int = 0
    tab_index: int = 0
    group_id: str | None = None


class FirefoxWindow(BaseModel):
    groups: list[FirefoxGroup] = Field(default_factory=list)
    tabs: list[FirefoxTab] = Field(default_factory=list)


class FirefoxSession(BaseModel):
    windows: list[FirefoxWindow] = Field(default_factory=list)

    @property
    def tabs(self) -> list[FirefoxTab]:
        return [
            tab
            for window in self.windows
            for tab in window.tabs
        ]

    @property
    def groups(self) -> list[FirefoxGroup]:
        return [
            group
            for window in self.windows
            for group in window.groups
        ]

    @property
    def tabs_count(self) -> int:
        return len(self.tabs)

    @property
    def windows_count(self) -> int:
        return len(self.windows)


def read_mozlz4_json(file_path: str | Path) -> dict[str, Any]:
    file_path = Path(file_path)
    raw = file_path.read_bytes()

    if not raw.startswith(MOZLZ4_MAGIC):
        raise ValueError(f'Not a Firefox mozlz4 file: {file_path}')

    data = lz4.block.decompress(raw[len(MOZLZ4_MAGIC):])
    return json.loads(data)


def read_firefox_session(file_path: str | Path) -> FirefoxSession:
    file_path = Path(file_path)
    obj = read_mozlz4_json(file_path)

    windows = obj.get('windows', [])
    global_index = 1
    window_models = []
    for window_index, window in enumerate(windows, start=1):
        groups = []
        for group_index, group in enumerate(window.get('groups', []), start=1):
            groups.append(FirefoxGroup(
                id=group.get('id'),
                name=group.get('name'),
                color=group.get('color'),
                collapsed=group.get('collapsed', False),
                save_on_window_close=group.get('saveOnWindowClose', True),
            ))

        tabs: list[FirefoxTab] = []
        for tab_index, tab in enumerate(window.get('tabs', []), start=1):
            entries = tab.get('entries') or []
            if not entries:
                continue

            # Firefox tab has `index`, 1-based current history entry index
            entry_index = tab.get('index', len(entries)) - 1
            if entry_index < 0 or entry_index >= len(entries):
                entry_index = len(entries) - 1

            entry = entries[entry_index]

            url = entry.get('url')
            if not url:
                continue

            title = entry.get('title')
            group_id = tab.get('groupId')

            last_accessed = tab.get('lastAccessed')
            if last_accessed:
                last_accessed = datetime.fromtimestamp(
                    last_accessed / 1000
                )

            favicon = tab.get('image')

            tabs.append(
                FirefoxTab(
                    index=global_index,
                    title=title,
                    url=url,
                    last_accessed=last_accessed,
                    favicon=favicon,
                    window_index=window_index,
                    tab_index=tab_index,
                    group_id=group_id
                )
            )
            global_index += 1

        window_models.append(FirefoxWindow(
            groups=groups,
            tabs=tabs
        ))

    return FirefoxSession(
        windows=window_models
    )


def extract_firefox_tabs(file_path: str | Path) -> list[FirefoxTab]:
    return read_firefox_session(file_path).tabs


def _format_dt(dt) -> str:
    if dt is None:
        return ''
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def firefox_session_to_markdown(session: FirefoxSession) -> str:
    parts = [
        '# Firefox Session',
        '',
        f'- Windows: {session.windows_count}',
        f'- Tabs: {session.tabs_count}',
        '',
    ]

    for window_index, window in enumerate(session.windows, start=1):
        parts.append(f'## Window {window_index}')
        parts.append('')

        groups_by_id = {
            group.id: group
            for group in window.groups
        }

        tabs_by_group: dict[str | None, list[FirefoxTab]] = {}

        for tab in window.tabs:
            tabs_by_group.setdefault(tab.group_id, []).append(tab)

        # 先输出有 group 的 tabs
        for group in window.groups:
            group_tabs = tabs_by_group.get(group.id, [])
            if not group_tabs:
                continue

            group_title = group.name or group.id
            parts.append(f'### {group_title}')
            parts.append('')

            for tab in group_tabs:
                title = tab.title or tab.url
                last_accessed = _format_dt(tab.last_accessed)

                if last_accessed:
                    parts.append(
                        f'- [{title}]({tab.url})  '
                        f'`{last_accessed}`'
                    )
                else:
                    parts.append(f'- [{title}]({tab.url})')

            parts.append('')

        # 再输出无 group 的 tabs
        ungrouped_tabs = tabs_by_group.get(None, [])

        if ungrouped_tabs:
            parts.append('### Ungrouped')
            parts.append('')

            for tab in ungrouped_tabs:
                title = tab.title or tab.url
                last_accessed = _format_dt(tab.last_accessed)

                if last_accessed:
                    parts.append(
                        f'- [{title}]({tab.url})  '
                        f'`{last_accessed}`'
                    )
                else:
                    parts.append(f'- [{title}]({tab.url})')

            parts.append('')

    return '\n'.join(parts).strip()


def firefox_session_to_txt(session: FirefoxSession) -> str:
    lines = [
        'index\twindow_index\ttab_index\tgroup\tlast_accessed\ttitle\turl'
    ]

    for window in session.windows:
        groups_by_id = {
            group.id: group
            for group in window.groups
        }

        for tab in window.tabs:
            group = (
                groups_by_id.get(tab.group_id)
                if tab.group_id
                else None
            )
            group_name = group.name if group else ''

            title = tab.title or ''
            last_accessed = _format_dt(tab.last_accessed)

            lines.append(
                '\t'.join(
                    [
                        str(tab.index),
                        str(tab.window_index),
                        str(tab.tab_index),
                        group_name,
                        last_accessed,
                        title,
                        tab.url,
                    ]
                )
            )

    return '\n'.join(lines)


def save_firefox_session_csv(
    session: FirefoxSession,
    output_file: str | Path,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open(
        'w',
        encoding='utf-8-sig',
        newline='',
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            'index',
            'window_index',
            'tab_index',
            'group_id',
            'group_name',
            'title',
            'url',
            'last_accessed',
            # 'favicon',
        ])

        for window in session.windows:
            groups_by_id = {
                group.id: group
                for group in window.groups
            }

            for tab in window.tabs:
                group = (
                    groups_by_id.get(tab.group_id)
                    if tab.group_id
                    else None
                )

                writer.writerow([
                    tab.index,
                    tab.window_index,
                    tab.tab_index,
                    tab.group_id,
                    group.name if group else '',
                    tab.title or '',
                    tab.url,
                    (
                        tab.last_accessed.isoformat()
                        if tab.last_accessed
                        else ''
                    ),
                    # tab.favicon or '',
                ])


def save_firefox_session_markdown(
    session: FirefoxSession,
    output_file: str | Path,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        firefox_session_to_markdown(session),
        encoding='utf-8',
    )


def save_firefox_session_txt(
    session: FirefoxSession,
    output_file: str | Path,
) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        firefox_session_to_txt(session),
        encoding='utf-8',
    )


if __name__ == '__main__':
    source_file = '/Users/andersc/Downloads/backups/ff-sessionstore-backups/upgrade.jsonlz4-20260525130955'
    session = read_firefox_session(source_file)
    # save_firefox_session_markdown(session, output_file='/Users/andersc/Downloads/ff-session-20260525130955.md')
    # save_firefox_session_txt(session, output_file='/Users/andersc/Downloads/ff-session-20260525130955.txt')
    save_firefox_session_csv(session, output_file='/Users/andersc/Downloads/ff-session-20260525130955.csv')
