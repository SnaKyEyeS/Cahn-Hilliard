#exCM3 Gaussian Antoine Quiriny
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift

sigma = 1
mu = pi

m = 100
N = 2*m

x = linspace(0, 2*pi, N)
f = (1/(sigma*sqrt(2*pi)))* exp(-(1/2)*((x-mu)**2/sigma**2))
df = (1/(sigma*sqrt(2*pi)))* exp(-(1/2)*((x-mu)**2/sigma**2)) * (-(x-mu)/sigma**2)

f_hat = fftshift(fft(f)/N)
k = arange(-N/2, N/2)

f_hat_dot = 1j*k*f_hat
f_dot = ifft(fftshift(f_hat_dot)).real *(2*pi/(2*pi/N))

plt.plot(x, df)
plt.plot(x, f_dot,".")
plt.legend(["Analytique derivative", "Spectral derivative"])
plt.show()
