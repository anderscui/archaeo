# coding=utf-8
from archaeo.chunking.base import split_long_text


def test_split_long_text():
    assert split_long_text('abcdefg', max_chars=3) == ['abc', 'def', 'g']
    assert split_long_text('hello world. test split.', max_chars=20) == ['hello world.', 'test split.']
    assert split_long_text('hello world. test split.', max_chars=25) == ['hello world. test split.']
