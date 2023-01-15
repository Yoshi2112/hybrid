import numba
import numpy as np
from numba import cuda
import cupy as cp

@cuda.jit
def calculate_charge_density(positions, charge, grid):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = positions.shape[0]
    for i in range(start, n, stride):
        x, y, z = positions[i]
        x, y, z = int(x), int(y), int(z)
        grid[x, y, z] += charge[i]

# Example usage
positions = cp.random.rand(100000, 3) * (255)
charge = cp.random.rand(100000)
grid = cp.zeros((256, 256, 256))

threadsperblock = 512
blockspergrid = (positions.shape[0] + (threadsperblock - 1)) // threadsperblock

calculate_charge_density[blockspergrid, threadsperblock](positions, charge, grid)

# copy density to host memory
density = cp.asnumpy(grid)
