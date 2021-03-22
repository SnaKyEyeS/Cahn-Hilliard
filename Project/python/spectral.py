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


def etdrk4(u, dt):
    """
    Time-stepping/integration scheme using the ETDRK4 scheme
    """
    # Non-linear term
    N = lambda u: k * (u - rfft2(irfft2(u)**3))

    # Linear term
    L = -(1e-2*k)**2

    # Pre-computations
    E = np.exp(L*dt)
    E2 = np.exp(L*dt/2.0)

    m = 32
    Q = np.zeros((n, int(n/2+1)))
    f1 = np.zeros((n, int(n/2+1)))
    f2 = np.zeros((n, int(n/2+1)))
    f3 = np.zeros((n, int(n/2+1)))

    for i in range(n):
        for j in range(int(n/2+1)):
            r = dt*L[i, j] + np.exp(1j*np.pi * (np.arange(m)+.5)/m)
            Q[i, j] = dt*np.mean(((np.exp(r/2) - 1) / r)).real
            f1[i, j] = dt*np.mean(((-4 - r + np.exp(r)*(4 - 3*r + r**2)) / r**3)).real
            f2[i, j] = dt*np.mean(((2 + r + np.exp(r)*(-2 + r)) / r**3)).real
            f3[i, j] = dt*np.mean(((-4 - 3*r - r**2 + np.exp(r)*(4 - r)) / r**3)).real

    while True:
        Nu = N(u)
        a = E2*u + Q*Nu
        Na = N(a)
        b = E2*u + Q*Na
        Nb = N(b)
        c = E2*a + Q*(2*Nb - Nu)
        Nc = N(c)
        u = E*u + f1*Nu + 2*f2*(Na + Nb) + f3*Nc
        yield u


def imex(c, dt):
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
sol = etdrk4(c, dt)

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
