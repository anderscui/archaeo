# coding=utf-8
from datetime import datetime, UTC

import arrow

from archaeo import logger


def str2int(s: str, default=0):
    if not s:
        return default
    try:
        return int(s.strip())
    except (ValueError, TypeError) as e:
        logger.error(f'str2int error: {e}')
        return default


def humanize_datetime(dt: datetime | str, locale='en'):
    arrow_dt = arrow.get(dt)
    return arrow_dt.humanize(locale=locale)


if __name__ == '__main__':
    assert str2int('12') == 12
    assert str2int('12.0') == 0
    assert str2int('abc') == 0
    assert str2int(None) == 0

    print(humanize_datetime(datetime.now(UTC)))
    print(humanize_datetime('2025-08-13T14:24:03Z'))
