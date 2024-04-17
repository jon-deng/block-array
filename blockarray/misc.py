"""
Miscellaneous function

Not sure where else to put these right now
"""

from typing import List, TypeVar, Any

T = TypeVar('T')


def replace(
    array: List[Any], keys: List[int], values: List[Any], copy: bool = True
) -> List[Any]:
    """
    Replace items in a list at specified indices

    The original list is mutated and returned.

    Parameters
    ----------
    array: List[T]
        The original list to perform the replacement on
    keys: List[int]
        The indices to replace values at
    values: List[T]
        The values used to replace
    copy: bool
        Whether to modify `array` in-place or return a copy

    Returns
    -------
    List[T]
        The same `array` with items replaced at given indices
    """
    if copy:
        ret_array = array.copy()
    else:
        ret_array = array

    for key, value in zip(keys, values):
        ret_array[key] = value
    return ret_array
