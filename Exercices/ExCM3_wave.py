import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft, fftshift
from scipy.integrate import solve_ivp as rk45


def du_dx(u):
    n = len(u)

    u_hat = fftshift(fft(u))
    k = np.arange(-n/2, n/2)

    u_dot_hat = 1j*k*u_hat
    u_dot = ifft(fftshift(u_dot_hat))

    return u_dot.real


def du_dt(t, u, x, c):
    return -c*du_dx(u)


if __name__ == "__main__":

    # Physical grid
    N = 2**8
    h = 2*np.pi/N
    x = np.arange(0, 2*np.pi, step=h)

    # c(x) - it is a constant...
    c = .2 + np.sin(x-1)**2

    # Time grid
    t_end = 8
    dt = h/4
    t = np.arange(0, t_end, step=dt)

    # Initial condition
    u_init = np.exp(-100*(x-1)**2)

    # Time stepping
    sol = rk45(du_dt, [0, t_end], u_init, t_eval=t, args=(x, c))

    # Plot results
    x_grid, t_grid = np.meshgrid(x, sol.t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, t_grid, sol.y.T, cmap="viridis")

    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, t_end)
    ax.set_zlim(0, 4)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Time axis")

    ax.view_init(30, 100)

    plt.show()
