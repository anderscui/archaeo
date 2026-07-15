# coding=utf-8
from urllib.parse import unquote_plus


def unquote_url(url: str):
    return unquote_plus(url)
