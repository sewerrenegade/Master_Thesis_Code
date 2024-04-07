"""Utilities module."""

from .general import is_iterable
from .general import nesting_level
from .general import pairwise
from .general import wrap_if_not_iterable

from .filters import SelectByDimension

from .summary_statistics import total_persistence
from .summary_statistics import persistent_entropy
from .summary_statistics import polynomial_function

__all__ = [
    "is_iterable",
    "nesting_level",
    "pairwise",
    "persistent_entropy",
    "polynomial_function",
    "total_persistence",
    "wrap_if_not_iterable",
    "SelectByDimension",
]