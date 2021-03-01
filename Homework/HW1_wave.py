from numpy import linspace, pi, sin, exp, meshgrid, round
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.fft import rfft, irfft, rfftfreq
from scipy.integrate import solve_ivp as rk45


# Physical grid
n = 2**9
x, h = linspace(0, 2*pi, n, endpoint=False, retstep=True)
c = .2 + sin(x-1)**2
u_init = .8*exp(-((x - pi) / .5)**2)
k = rfftfreq(n, h/(2*pi))

# Solve EDP using RK45 time integration
t_end = 30
sol = rk45(lambda _, u: -c*irfft(1j*k*rfft(u)), (0, t_end), u_init, rtol=1e-6, atol=1e-8)

# Plot results
# Two modes are possible:
#  - A continuous animation (can be saved)
#  - A 2D plot with time axis
animation = True

if animation:
    fig = plt.figure()
    ax = plt.axes(xlim=(0, x[-1]), ylim=(0, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    line, = ax.plot([], [], lw=3)
    title = ax.text(.5, .9, "", bbox={'facecolor': 'w', 'alpha': 0.7, 'pad': 5}, transform=ax.transAxes, ha="center")

    def update(i):
        line.set_data(x, sol.y[:, i])
        title.set_text(f"Time = {sol.t[i]:.2f} [s]")
        return line, title,

    anim = FuncAnimation(fig, update, frames=len(sol.t), interval=t_end/len(sol.t)*8e2, blit=True)
    plt.show()

    # To save the animation if needed
    # from tqdm import tqdm
    # pbar = tqdm(total=len(sol.t))
    # anim.save("wave.mp4", progress_callback=lambda i, n: pbar.update(1), fps=len(sol.t)/t_end)

else:
    idx = round(linspace(0, len(sol.t)-1, 20)).astype(int)
    x_grid, t_grid = meshgrid(x, sol.t[idx])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(x_grid, t_grid, sol.y[:, idx].T, cstride=0)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(0, t_end)
    ax.set_zlim(0, 2)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Time axis")
    ax.view_init(30, -80)
    plt.show()
