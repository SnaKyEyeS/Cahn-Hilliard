import sys
import numpy as np
import scipy.io as io

from copy import copy
from spectral import integrate, irfft2, rfft2, n


dts = np.array([1e-6, 1e-7])
error = list()
np.random.seed(seed=1)
c = rfft2(2*np.random.rand(n, n) - 1)
ref = irfft2(io.loadmat("sol_ref.mat")["c"])

for dt in dts:
    # Time scheme - imex & rk4 as ref
    imex = integrate(copy(c), dt)
    iter = 0

    # Iterate to t = 0.001
    print(f"\rStarting iterations for dt = {dt*1e3:.4f} [ms]...")
    while not dt*iter == .001:
        sol = next(imex)

        sys.stdout.write(f"\rt = {dt*iter:.6f} [s]")
        iter += 1
    error.append(irfft2(sol))

print(f"\rn_est = {np.log10(np.sqrt(np.sum((error[0] - ref)**2, axis=(0, 1))) / np.sqrt(np.sum((error[1] - ref)**2, axis=(0, 1))))}")
