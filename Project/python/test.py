import sys
import numpy as np

from copy import copy
from spectral import integrate, irfft2, rfft2, n


# Load initial & reference solutions
# Reference solution is taken at t = 1e-3 w/ RK4 using dt = 1e-9
c_init = rfft2(np.loadtxt("c_128_init.txt").reshape(n, n))
c_ref = np.loadtxt("c_128_ref.txt").reshape(n, n)

# Initialise stuff
dts = [1e-6/4, 1e-6/8]
error = list()

for dt in dts:
    # Time scheme - imex & rk4 as ref
    imex = integrate(copy(c_init), dt)
    iter = 0

    # Iterate to t = 0.001
    print(f"\rStarting iterations for dt = {dt*1e3:.4f} [ms]...")
    while not dt*iter == .001:
        sol = next(imex)

        sys.stdout.write(f"\rt = {dt*iter:.6f} [s]")
        iter += 1
    error.append(np.sqrt(np.sum((irfft2(sol) - c_ref)**2)))

print(f"\rn_est = {np.log(error[0] / error[1]) / np.log(dts[0]/dts[1])}")
