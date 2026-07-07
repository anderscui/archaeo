# coding=utf-8
from collections import Counter
from typing import Sequence

import pytest

from archaeo.iterable import find_by, find_last_by, find_max, rename_keys, flatten_list
from archaeo.iterable import find_key_by_value, first_or, is_seq_of_type, is_seq_of_str, filter_counter


def test_find_by():
    actual = find_by([], pred=lambda c: c == 'e')
    assert actual == -1

    actual = find_by('hello', pred=lambda c: c == 'e')
    assert actual == 1


def test_find_last_by():
    actual = find_last_by([], pred=lambda c: c == 'e')
    assert actual == -1

    actual = find_last_by('hello', pred=lambda c: c == 'l')
    assert actual == 3


@pytest.mark.parametrize(
    ('iterable', 'key', 'expected'),
    [
        ([], None, (-1, None)),
        ([1, 2, 3], None, (2, 3)),
        ('hello, world', None, (7, 'w')),
        (['hello, world', 'cool', 'python'], len, (0, 'hello, world')),
    ]
)
def test_find_max(iterable, key, expected):
    actual = find_max(iterable, key) if key else find_max(iterable)
    assert actual == expected


def test_rename_keys():
    d = {'a': 1, 'b': 2}
    renamed = rename_keys(d, {'a': 'c'})
    assert list(renamed.keys()) == ['c', 'b']


def test_flatten_list():
    items = [[1, 2, 3, 4], [5, 6, 7], [8, 9]]
    flattened = flatten_list(items)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flattened == expected

    items = [[1, 2, 3, 4], 5, [6, 7], 8, [9]]
    flattened = flatten_list(items)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flattened == expected

    items = [[1, [2, 3], 4], [5, [6], 7], 8, [9]]
    flattened = flatten_list(items)
    expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert flattened == expected


def test_find_key_by_value():
    d = {}
    expected = None
    assert expected == find_key_by_value(d, pred=lambda v: v == 1)

    d = {1: 'a', '2': 'b'}
    expected = 1
    assert expected == find_key_by_value(d, pred=lambda v: v == 'a')

    d = {1: (1, 2), 2: (3, 4), 3: (5, 6)}
    expected = 3
    assert expected == find_key_by_value(d, pred=lambda v: v[0] <= 5.5 <= v[1])


def test_first_or():
    d = {}
    assert first_or(d, None) is None
    assert first_or(d.items(), None) is None

    d = 'abc'
    assert 'a' == first_or(d)

    d = range(5, 10)
    assert 5 == first_or(d)

    d = []
    assert first_or(d) is None


def test_is_seq_of_type():
    s = []
    assert is_seq_of_type(s, int)

    s = [1, 2, 3]
    assert is_seq_of_type(s, int)

    s = 'hello, world'
    assert not is_seq_of_type(s, int)

    s = ('hello, world', 'cool')
    assert not is_seq_of_type(s, float)
    assert is_seq_of_type(s, Sequence)


def test_is_seq_of_str():
    s = []
    assert is_seq_of_str(s)

    s = [1, 2, 3]
    assert not is_seq_of_str(s)

    s = 'hello, world'
    assert is_seq_of_str(s)

    s = ['hello, world', 'cool']
    assert is_seq_of_str(s)

def test_filter_counter():
    c = Counter({'a': 1, 'b': 2, 'c': 3})

    c2 = filter_counter(c, lambda item: item[1] > 1)
    expected = Counter({'b': 2, 'c': 3})
    assert c2 == expected

    c2 = filter_counter(c, lambda item: item[1] > 10)
    expected = Counter()
    assert c2 == expected
