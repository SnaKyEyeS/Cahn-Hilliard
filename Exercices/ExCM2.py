import warnings

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import rfft, irfft, rfftfreq


def diff(func, start, end, N, n=None):
    """
    Return the 1-D derivative of f, of specified order.
    """
    # Sample function f
    L = end - start
    h = L/N
    x = np.arange(start, end, step=h)
    f = func(x)

    # # Over/under sampling
    if n is None:
        n = int(N)
    else:
        n = int(n)
        h = L/n
        x = np.arange(start, end, step=h)

    # Fourier Transform
    f_hat = rfft(f)

    # Wavelength
    k = rfftfreq(N, h/(2*np.pi))

    # Derive f_hat
    f_hat_dot = 1j*k*f_hat

    # Inverse Fourier Transform
    f_dot = irfft(f_hat_dot, n=n).real

    return x, f_dot


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
    x, f_dot = diff(rect, -2*np.pi, 2*np.pi, 10000000, order=1)

    plt.plot(x, f_dot)
    plt.plot(x, f(x)*np.cos(x))
    # plt.plot(x, f(x)*(np.cos(x)**2 - np.sin(x)))
    plt.show()
