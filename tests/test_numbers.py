# coding=utf-8
import pytest

from archaeo.numbers import is_perfect_number


@pytest.mark.parametrize(
    ('n', 'expected'),
    [
        (-1, False),
        (0, False),
        (1, False),
        (2, False),
        (6, True),
        (12, False),
        (28, True),
        (496, True),
    ]
)
def test_is_perfect_number(n, expected):
    assert is_perfect_number(n) == expected


def test_is_perfect_number_series():
    first_perfect_numbers = [6, 28, 496]
    # first_perfect_numbers = [6, 28, 496, 8128, 33550336]
    actual = [i for i in range(-1, first_perfect_numbers[-1] + 1) if is_perfect_number(i)]
    assert actual == first_perfect_numbers
