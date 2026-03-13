"""
py_contact_ms — Contact Molecular Surface for Python

Ported from Longxing Cao's C++ implementation. Based on Lawrence & Colman (1993).
"""

__version__ = "0.1.0"

from py_contact_ms._core import (
    calculate_contact_ms,
    get_radii_from_names,
    calculate_maximum_possible_contact_ms,
    partition_pose,
)

__all__ = [
    "calculate_contact_ms",
    "get_radii_from_names",
    "calculate_maximum_possible_contact_ms",
    "partition_pose",
]
