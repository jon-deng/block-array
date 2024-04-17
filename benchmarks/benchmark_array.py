"""
Contains functions for benchmarking `BlockArray` functions

Run this using `cProfile`, `line_profiler`, etc. as an entry point to
benchmarking
"""

import numpy as np

from blockarray.blockarray import BlockArray


def setup_subarrays():
    sizes = (500, 500, 100, 9000)
    subarrays = [np.ones(size) for size in sizes]
    _subarrays = np.empty(len(subarrays), dtype=object)
    _subarrays[:] = subarrays
    return _subarrays


def benchmark_array_creation(subarrays):
    return BlockArray(subarrays, labels=(('a', 'b', 'c', 'd'),))


if __name__ == '__main__':
    subarrays = setup_subarrays()

    for n in range(2000):
        benchmark_array_creation(subarrays)
