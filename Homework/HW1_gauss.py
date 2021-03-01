from numpy import linspace, pi, exp, geomspace, zeros, sqrt, abs
import matplotlib.pyplot as plt

from scipy.fft import fft, ifft, fftfreq, fftshift


# Figures
fig, axs = plt.subplots(2)
fig.suptitle("Frequency spectrum & differentiation error")

# Physical grid & stuff
n = 2**12
x, h = linspace(0, 2*pi, n, endpoint=False, retstep=True)
k = fftfreq(n, h/(2*pi))
f  = lambda x, sigma: exp(-.5*(x-pi)**2/sigma**2) / (sigma*sqrt(2*pi))
df = lambda x, sigma: -f(x, sigma)*(x-pi) / sigma**2

# Compare spectrum for different Gaussiam "width"
sigmas = geomspace(.001, .1, 3)
for sigma in sigmas:
    gauss = f(x, sigma)
    gauss_hat = fftshift(fft(gauss)) / n
    axs[0].plot(fftshift(k), abs(gauss_hat), label=f"Sigma = {sigma:.3f}")

    gauss_dot = ifft(1j*k*fft(gauss)).real
    error = abs(gauss_dot - df(x, sigma)).mean()
    axs[1].loglog(sigma, error, 'D', label=f"Sigma = {sigma:.3f}")


axs[0].set_xlabel("k")
axs[0].set_ylabel("Freq. spectrum magnitude")
axs[0].legend()

# Compare error for different Gaussian "width"
sigmas = geomspace(.001, .1, 100)
errors = zeros(sigmas.shape)
for i, sigma in enumerate(sigmas):
    gauss = f(x, sigma)
    gauss_dot = ifft(1j*k*fft(gauss)).real
    errors[i] = abs(gauss_dot - df(x, sigma)).mean()

axs[1].loglog(sigmas, errors, '--', zorder=-1)
axs[1].set_xlabel("Sigma")
axs[1].set_ylabel("Differentiation error")
axs[1].legend()

plt.show()
