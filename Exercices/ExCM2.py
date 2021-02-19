import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift


def diff(func, start, end, N, n=None):
    """
    Return the 1-D derivative of f, of specified order.
    """
    # Sample function f
    L = end - start
    h = L/N
    x = np.arange(start, end, step=h)
    f = func(x)

    # Over/under sampling
    if n is None:
        n = int(N)
    else:
        n = int(n)
        h = L/n
        x = np.arange(start, end, step=h)

    # Compute FFT
    f_hat = fftshift(fft(f)/N)
    k = np.arange(-N/2, N/2)

    # Compute derivative
    f_dot_hat = (1j*k) * f_hat
    f_dot = ifft(fftshift(f_dot_hat), n=n)

    return x, f_dot.real * (2*np.pi/h)


def rect(x, lim=1):
    """
    Rectangular function
    """
    y = np.zeros(x.shape)
    y[(x > -lim) & (x < lim)] = 1

    if np.any(abs(x) == lim):
        warnings.warn("WARNING: Sampling rectangular function at discontinuity !")

    return y


if __name__ == "__main__":

    # Some function definition
    f = lambda x: np.exp(np.sin(x))

    # Derive function
    x, f_dot = diff(f, 0, 2*np.pi, 64, 128)

    plt.plot(x, f_dot)
    plt.plot(x, f(x)*np.cos(x))
    # plt.plot(x, f(x)*(np.cos(x)**2 - np.sin(x)))
    plt.show()
