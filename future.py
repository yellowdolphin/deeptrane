# PEP616, version 3.9
from typing import TypeVar

T = TypeVar('T', str, bytes, bytearray)


def removeprefix(s: T, prefix: T) -> T:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s[:]


def removesuffix(s: T, suffix: T) -> T:
    if suffix and s.endswith(suffix):
        return s[:-len(suffix)]
    return s[:]
