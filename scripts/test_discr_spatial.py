import sys
import numpy as np
import matplotlib

from copy import copy
from scipy.linalg import norm
from spectral import irfft2, rfft2, n
from spectral import etdrk4 as integrate


np.random.seed(0)
dt = 1e-6

# Initialise stuff
x, y = np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n))


# # Intializing sigma and muu
# dst = np.sqrt(x*x+y*y)
# sigmax = 0.5
# sigmay = 0.5
# mux = 0.5
# muy = 0.5
#
# # Calculating Gaussian array
# gauss = np.exp(-( (x-mux)**2 / ( 2.0 * sigmax**2 ) + (y-muy)**2 / ( 2.0 * sigmay**2 ) ) )
# c_init = rfft2(gauss)
unit = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        if (np.abs(x[0,i]) <0.75 and np.abs(x[0,i]) > 0.25) and (np.abs(y[j,0]) <0.75 and np.abs(y[j,0]) > 0.25) :
            unit[i,j] = 1

c_init = rfft2(unit)



# Time scheme - imex & rk4 as ref
step = integrate(copy(c_init), dt)
iter = 0

# Iterate to t = 0.001
print(f"\rStarting iterations for dt = {dt*1e3:.4f} [ms]...")
while not dt*iter == .001:
    c = next(step)

    sys.stdout.write(f"\rt = {dt*iter:.6f} [s]")
    iter += 1

c = irfft2(c)



np.savetxt('spatial_discr_results/unit/n_2_etdrk4.txt', c, fmt='%.10e')
