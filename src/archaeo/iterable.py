# coding=utf-8
from collections import Counter
from collections.abc import Iterable, Callable, Sequence
from copy import deepcopy
from difflib import SequenceMatcher
from itertools import islice
from typing import Any, TypeVar

T = TypeVar("T")


def first(iterable):
    """ The first element in a iterable"""
    return next(iter(iterable))


def find_by(iterable: Iterable[T],
            pred: Callable[[T], bool],
            start=0) -> int:
    for i, elem in enumerate(islice(iterable, start, None), start=start):
        if pred(elem):
            return i
    return -1


def find_last_by(iterable: Sequence[T],
                 pred: Callable[[T], bool]) -> int:
    for i, elem in enumerate(reversed(iterable)):
        if pred(elem):
            return len(iterable) - 1 - i
    return -1


def find_max(iterable: Iterable[T],
             key: Callable[[T], Any] = lambda x: x) -> tuple[int, T | None]:

    max_i = -1
    max_val: T | None = None
    max_key: Any = None

    for i, elem in enumerate(iterable):
        elem_key = key(elem)

        if max_i == -1 or elem_key > max_key:
            max_i = i
            max_key = elem_key
            max_val = elem

    return max_i, max_val


def find_key_by(d: dict, pred):
    for key in d.keys():
        if pred(key):
            return key
    return None


def find_key_by_value(d: dict, pred):
    for k, v in d.items():
        if pred(v):
            return k
    return None


def subset_of(s1, s2):
    """
    Determine whether `s1` is a subset of `s2`.
    :param s1:
    :param s2:
    """
    s2 = set(s2)
    return all(item in s2 for item in s1)


def drop_dup(iterable, key=None):
    if key is None:
        key = lambda _item: _item

    found = set()
    result = []
    for item in iterable:
        item_key = key(item)
        if item_key not in found:
            result.append(item)
            found.add(item_key)
    return result


def rename_keys(d: dict, key_map: dict):
    return {key_map.get(k, k): v for k, v in d.items()}


def filter_by_keys(d: dict | list[dict], keys):
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if k in keys}
    elif isinstance(d, (list, tuple)):
        return [{k: v for k, v in item.items() if k in keys} for item in d]
    raise TypeError(f'Unsupported type: {type(d)}')


def pop_keys(d: dict | list[dict], keys: Sequence):
    if isinstance(d, dict):
        return {k: v for k, v in d.items() if k not in keys}
    elif isinstance(d, (list, tuple)):
        return [{k: v for k, v in item.items() if k not in keys} for item in d]
    raise TypeError(f'Unsupported type: {type(d)}')


def get_dict_values(d: dict, keys, strict=True):
    if strict:
        return [d[key] for key in keys]
    return [d.get(key) for key in keys]


def set_value_by_key_path(d: dict, key_path, v):
    result = deepcopy(d)

    parts = key_path.split('.')
    cur = result
    for key_path in parts[:-1]:
        is_array = key_path.endswith('[]')
        if is_array:
            key_path = key_path[:-2]

        if key_path not in cur:
            if is_array:
                cur[key_path] = [{}]
            else:
                cur[key_path] = {}

        if is_array:
            cur = cur[key_path][0]
        else:
            cur = cur[key_path]

    cur[parts[-1]] = v
    return result


def map_dict(d: dict, mapping: dict):
    new_d = {}
    for k, v in d.items():
        new_d[mapping.get(k, k)] = v
    return new_d


def dicts_equal(d1: dict, d2: dict):
    if d1 is None and d2 is None:
        return True
    if d1 is None or d2 is None:
        return False

    if len(d1) != len(d2):
        return False

    for k, v in d1.items():
        if k not in d2 or d2[k] != v:
            return False
    return True


def most_common_key(c: Counter):
    if c:
        return c.most_common(1)[0][0]
    return None


def top_of_counter(c: Counter, ratio=1.0):
    total = c.total()
    tops = []
    n = 0
    for k, cnt in c.most_common():
        tops.append((k, cnt))
        n += cnt
        if n / total >= ratio:
            break
    return tops


def filter_counter_by_key(c: Counter, key_pred):
    new_c = Counter()
    for k, cnt in c.items():
        if key_pred(k):
            new_c[k] += cnt

    return new_c


def filter_counter_by_value(c: Counter, value_pred) -> Counter:
    return Counter({k: cnt for k, cnt in c.items() if value_pred(cnt)})


def filter_counter(c: Counter, pred) -> Counter:
    return Counter({item[0]: item[1] for item in c.items() if pred(item)})


def seq_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def first_or(iterable: Iterable, default: Any = None):
    if iterable:
        return first(iterable)
    return default


def is_seq_of_type(items: Sequence[Any], t: type):
    return all(isinstance(item, t) for item in items)


def is_seq_of_str(items: Sequence[Any]):
    return is_seq_of_type(items, str)


def flatten_list(items: list):
    flattened = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


