from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def update(i):
    """
    Animation update function
    """
    for _ in range(skip_frame):
        c = next(sol)

    img.set_data(c)
    title.set_text(f"Time = {i*skip_frame*dt:.6f}")
    return img, title,


def integrate(c):
    """
    Time-stepping/integration scheme
    """
    # Explicit Euler
    laplacian = lambda c: (np.roll(c, 1, axis=0) + np.roll(c, -1, axis=0) + np.roll(c, 1, axis=1) + np.roll(c, -1, axis=1) - 4*c) / h**2
    while True:
        c += dt*laplacian(c**3 - c - a**2*laplacian(c))
        yield c


# Problem parameters
a = 1e-2
n = 128
dt = 1e-6
n_step = 12000
skip_frame = 10

# Initialise vectors
x, h = np.linspace(0, 1, n, endpoint=False, retstep=True)
c = 2*np.random.rand(n, n) - 1
sol = integrate(c)

# Initialize animation
fig, ax = plt.subplots()
img = ax.imshow(c, cmap="jet", vmin=-1, vmax=1)
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
