import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from spectral import irfft2, rfft2


N = [2, 4, 8, 16, 32, 64, 128, 256] #[2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]

C_ref = np.loadtxt("spatial_discr_results/gaussian/n_512_etdrk4.txt")

error = list()

for n in N:
    C = np.loadtxt("spatial_discr_results/gaussian/n_" + str(int(n)) + "_etdrk4.txt")
    step = int(512/n)
    error.append(norm( C - C_ref[0:511:step, 0:511:step]) / norm(C_ref[0:511:step, 0:511:step]))

    #irfft2(rfft2(C), s=(512,512))

plt.plot(N, error, 'o')
plt.yscale("log")
plt.show()
