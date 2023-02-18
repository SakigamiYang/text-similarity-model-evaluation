# coding: utf-8
from pathlib import Path

__all__ = [
    'Directories'
]

_root = Path(__file__).parent.parent


class Directories:
    DATA = _root / 'data'
    DATA_SIMCLUE = DATA / 'simclue_public'
