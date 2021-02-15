import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift


def diff(func, start, end, N, order=1):
    """
    Return the 1-D derivative of f, of specified order.
    """
    L = end - start
    h = L/N
    x = np.arange(start+h, end+h, step=h)

    # Compute FFT
    f = func(x)
    f_hat = fftshift(fft(f))
    k = np.arange(-N/2, N/2)

    # Compute derivative
    f_dot_hat = (1j*k)**order * f_hat
    f_dot = ifft(fftshift(f_dot_hat))

    return x, f_dot.real * (2*np.pi/L)**order


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
    x, f_dot = diff(rect, -2*np.pi, 2*np.pi, 1e3, order=1)

    plt.plot(x, f_dot)
    # plt.plot(x, f(x)*np.cos(x))
    # plt.plot(x, f(x)*(np.cos(x)**2 - np.sin(x)))
    plt.show()
