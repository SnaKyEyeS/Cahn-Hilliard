from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq


def update(i):
    """
    Animation update function
    """
    for _ in range(skip_frame):
        c = next(sol)

    img.set_data(irfft2(c))
    title.set_text(f"Time = {i*skip_frame*dt:.6f}")
    return img, title,


def integrate(c, dt):
    """
    Time-stepping/integration scheme using a semi-implicit scheme combining
    the Backward-Differentiation Formulae & Adams-Bashforth
    """
    f = lambda c: rfft2(irfft2(c)**3) - c

    # First step
    c_prev = c
    f_prev = f(c)
    c -= dt*k*f_prev
    c /= (1 + dt*(a*k)**2)
    yield c

    while True:
        c_curr = c
        f_curr = f(c_curr)
        c = 4*c_curr - c_prev - 2*dt*k*(2*f_curr - f_prev)
        c /= (3 + 2*dt*(a*k)**2)
        c_prev = c_curr
        f_prev = f_curr
        yield c


def rk4(c, dt):
    """
    Time stepping/integration scheme using the classical 4th order Runge-Kutta
    """
    f = lambda c: -k*(rfft2(irfft2(c)**3) - c + a**2*k*c)
    while True:
        k1 = f(c)
        k2 = f(c + dt*k1/2)
        k3 = f(c + dt*k2/2)
        k4 = f(c + dt*k3)

        c += dt*(k1 + 2*k2 + 2*k3 + k4)/6
        yield c


# Problem parameters
a = 1e-2
n = 128
dt = 1e-6
n_step = 12000*4
skip_frame = 10

# Initialise vectors
x, h = np.linspace(0, 1, n, endpoint=False, retstep=True)
c = rfft2(2*np.random.rand(n, n) - 1)   # We work in frequency domain
sol = integrate(c, dt)

# Initialize wavelength for second derivative to avoid a repetitive operation
# Since we use rfftn, one dim is n/2+1 (rfftfreq) and the other is n (fftfreq)
k_x, k_y = np.meshgrid(rfftfreq(n, h/(2*np.pi)), fftfreq(n, h/(2*np.pi)))
k = k_x**2 + k_y**2

# Initialize animation
if __name__ == "__main__":
    fig, ax = plt.subplots()
    img = ax.imshow(irfft2(c), cmap="jet", vmin=-1, vmax=1)
    fig.colorbar(img, ax=ax)
    ax.axis("off")
    title = ax.text(.5, .1, "", bbox={'facecolor': 'w', 'alpha': 0.7, 'pad': 5}, transform=ax.transAxes, ha="center")

    # Start animation
    anim = FuncAnimation(fig, update, frames=int(n_step/skip_frame), interval=1, blit=True)
    if False:
        pbar = tqdm(total=int(n_step/skip_frame))
        anim.save("cahn_hilliard_spectral.mp4", fps=500, progress_callback=lambda i, n: pbar.update(1))
    else:
        plt.show()
