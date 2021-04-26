from spectral import rk4 as integrate
from spectral import irfft2, rfft2, n

import numpy as np
import time
from pypapi import events, papi_high as high

dt= 1e-6

c = rfft2(2*np.random.rand(n, n) - 1)   # We work in frequency domain
sol = integrate(c, dt)

start = time.time()
# high.start_counters([events.PAPI_DP_OPS,])
#
for _ in range(1000):
    c = next(sol)
#
# x=high.stop_counters()
#
# print(x)
end = time.time()
print(end-start)

#format data_time: Python N = 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024
#                  C
#                  Cuda
