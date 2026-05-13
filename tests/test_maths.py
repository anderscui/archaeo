# coding=utf-8
import pytest

from archaeo.maths import are_close, factorial


def test_are_close():
    assert not are_close(4.2030029296875, 3.8790283203125, 0.1)
    assert are_close(3.11, 3.13, 0.1)


@pytest.mark.parametrize(
    ('n', 'expected'),
    [
        (0, 1),
        (1, 1),
        (2, 2),
        (3, 6),
        (4, 24),
        (5, 120),
    ]
)
def test_factorial(n, expected):
    assert factorial(n) == expected
