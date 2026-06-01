# coding=utf-8
import re

RE_ZH_ALPHABET = re.compile(r'[A-Za-z\u4e00-\u9fff]', re.U)


def contains_zh_or_alphabet(s):
    return RE_ZH_ALPHABET.search(s) is not None
