# coding=utf-8


def rename_keys(d: dict, key_map: dict):
    return {key_map.get(k, k): v for k, v in d.items()}
