# coding=utf-8
from archaeo.iterable import rename_keys


def test_rename_keys():
    d = {'a': 1, 'b': 2}
    renamed = rename_keys(d, {'a': 'c'})
    assert list(renamed.keys()) == ['c', 'b']
