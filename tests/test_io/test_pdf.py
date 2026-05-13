# coding=utf-8
from datetime import datetime, timezone, timedelta

import pytest

from archaeo.io.pdf import parse_pdf_date


@pytest.mark.parametrize(
    ('date_str', 'expected'),
    [
        ('D:20260510', datetime(year=2026, month=5, day=10)),
        ('D:20260510212950', datetime(year=2026, month=5, day=10, hour=21, minute=29, second=50)),
        ('D:20260510212950Z', datetime(year=2026, month=5, day=10, hour=21, minute=29, second=50, tzinfo=timezone.utc)),
        ("D:20260510212950+08'00'", datetime(year=2026, month=5, day=10, hour=21, minute=29, second=50,
                                             tzinfo=timezone(timedelta(seconds=28800)))),
        ("D:20260510212950-03'00'", datetime(year=2026, month=5, day=10, hour=21, minute=29, second=50,
                                             tzinfo=timezone(timedelta(seconds=-10800)))),
    ]
)
def test_parse_pdf_date(date_str, expected):
    assert parse_pdf_date(date_str) == expected
