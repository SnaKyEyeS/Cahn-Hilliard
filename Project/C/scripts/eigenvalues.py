import numpy as np
from scipy.fft import fftfreq, fft2
from matplotlib import pyplot as plt


# Load reference solutions
# Reference solution is taken at t = 1e-3 w/ RK4 using dt = 1e-9
ref = np.loadtxt("rk4_128_1e-9.txt")

# Plot eigenvalues
n = 128
x, h = np.linspace(0, 1, n, endpoint=False, retstep=True)
k_x, k_y = np.meshgrid(fftfreq(n, h/(2*np.pi)), fftfreq(n, h/(2*np.pi)))
k = np.sqrt(k_x**2 + k_y**2)
c_hat_2 = fft2(ref.reshape(n, n)**2) / n**2
eig = -k**2 * (3*c_hat_2 - 1) - 1e-4*k**4
eig = eig.reshape(n*n)


plt.rcParams.update({"text.usetex": True})
plt.scatter(eig.real, eig.imag, s=0.2)
plt.xlabel("$\\Re(\\lambda)$")
plt.ylabel("$\\Im(\\lambda)$")
plt.show()
# plt.savefig("eigenvalues.pdf")
