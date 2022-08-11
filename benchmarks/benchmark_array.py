import cProfile

import numpy as np

from blockarray.blockarray import BlockArray

def setup_subarrays():
    sizes = (500, 500, 100, 9000)
    subarrays = [np.ones(size) for size in sizes]
    return subarrays

def benchmark_array_creation(subarrays):
    return BlockArray(subarrays, labels=(('a', 'b', 'c', 'd'),))

if __name__ == '__main__':
    subarrays = setup_subarrays()

    cProfile.run('for n in range(2000): benchmark_array_creation(subarrays)', 'benchmark_array_creation.Profile')
