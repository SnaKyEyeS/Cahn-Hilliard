import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fftpack import fftn, ifftn, fftshift


def init():
    """
    Animation initialization function
    """
    c[:, :] = 2*np.random.rand(n, n) - 1

    img.set_data(c)
    title.set_text("Time = 0")
    return img, title,


def update(i):
    """
    Animation update function
    """
    img.set_data(next(sol))
    title.set_text(f"Step = {i}/{n_step}")
    return img, title,


def laplacian(c):
    """
    Compute the Laplacian of c
    """
    c_hat = fftshift(fftn(c))
    k = np.arange(-n/2, n/2)
    k_x, k_y = np.meshgrid(k, k)
    c_hat_dot = -c_hat * (k_x**2 + k_y**2) * (2*np.pi/(n*h))**2
    return ifftn(fftshift(c_hat_dot)).real


def f(c):
    """
    Compute the derivative dc_dt = f(c)
    """
    return laplacian(c**3 - c - a**2*laplacian(c))


def integrate(c):
    """
    Time-stepping/integration scheme
    """
    # Runge-Kutta 4
    while True:
        k1 = f(c)
        k2 = f(c + dt*k1/2)
        k3 = f(c + dt*k2/2)
        k4 = f(c + dt*k3)
        c = c + dt*(k1 + 2*k2 + 2*k3 + k4)/6

        yield c


# Problem parameters
a = 1e-2
n = 128
dt = 1e-6/4
n_step = 12000*4

# Initialise vectors
x, h = np.linspace(0, 1, n, endpoint=False, retstep=True)
c = np.zeros((n, n))
sol = integrate(c)

# Initialize animation
fig, ax = plt.subplots()
img = ax.imshow(c, cmap="jet", vmin=-1, vmax=1)
ax.axis("off")
title = ax.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.7, 'pad': 5}, transform=ax.transAxes, ha="center")

# Start animation
# anim.save("cahn_hilliard_spectral.gif", writer="imagemagick")
anim = FuncAnimation(fig, update, init_func=init, frames=n_step, interval=1, blit=True)
plt.show()
