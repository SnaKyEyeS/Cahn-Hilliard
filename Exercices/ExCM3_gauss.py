import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift


# Initialize subplots
# fig, axs = plt.subplot(2, 2)

# Physical grid
start = -2*np.pi
end = 2*np.pi
N = 2**6
h = (end-start)/N
x = np.arange(start, end, step=h)

# Gaussian function
mu = 0
sigma = 1
f  = lambda x: np.exp(-.5*(x-mu)**2/sigma**2) / (sigma*np.sqrt(2*np.pi))
df = lambda x: -f(x)*(x-mu) / sigma**2

# Discrete Fourier Transform
f_hat = fftshift(fft(f(x))/N)
k = np.arange(-N/2, N/2)

# Take derivative
f_dot_hat = 1j*k*f_hat
f_dot = ifft(fftshift(f_dot_hat)).real * (2*np.pi/h)

x_discr = np.linspace(start, end, 10000)
plt.plot(x_discr, df(x_discr))
plt.plot(x, f_dot, ".r")
plt.legend(["Exact", "Spectral"])

plt.show()
