import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from scipy.fft import rfft2, irfft2, fftfreq, rfftfreq


def update(i):
    """
    Animation update function
    """
    for _ in range(skip_frame):
        c = next(sol)

    img.set_data(irfft2(c))
    title.set_text(f"Time = {i*skip_frame*dt:.6f}")
    return img, title,


def non_linear_term(c_hat):

    c = irfft2(c_hat)
    c[c > 1] = 1
    c[c < -1] = -1
    f_hat = rfft2(c**3 - c)

    m = 1.0 - a*c**2 - A

    vec_x = rfft2(irfft2(1j*k_x*(f_hat + kappa*k*c_hat)) * m)
    vec_y = rfft2(irfft2(1j*k_y*(f_hat + kappa*k*c_hat)) * m)

    return 1j*k_x*vec_x + 1j*k_y*vec_y - A*k*f_hat


def etdrk4(u, dt):
    """
    Time-stepping/integration scheme using the ETDRK4 scheme
    """
    # Non-linear term
    N = lambda u: non_linear_term(u)

    # Linear term
    L = -A*kappa*k**2

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

    # Loop !
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


# Problem parameters
A = .5
a = 1
kappa = 1
n = 128
dt = 1
n_step = 12000
skip_frame = 10

# Initialise vectors
x, h = np.linspace(0, n, n, endpoint=False, retstep=True)
c = rfft2(.2*np.random.rand(n, n) - .1)   # We work in frequency domain
sol = etdrk4(c, dt)

# Initialize wavelength for second derivative to avoid a repetitive operation
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
