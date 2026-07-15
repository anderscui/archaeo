# coding=utf-8
import os


def set_local_proxies(provider='v2ray'):
    if provider == 'crown':
        os.environ["http_proxy"] = "http://127.0.0.1:1235"
        os.environ["https_proxy"] = "http://127.0.0.1:1235"
    elif provider == 'v2ray':
        os.environ["http_proxy"] = "http://127.0.0.1:10808"
        os.environ["https_proxy"] = "http://127.0.0.1:10808"
    else:
        raise ValueError('Unknown proxy provider.')
