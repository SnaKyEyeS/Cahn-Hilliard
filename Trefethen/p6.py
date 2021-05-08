from numpy import linspace, pi, sin, arange, zeros, meshgrid, exp
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq


# Grid & initial data
n = 128
x, h = linspace(0, 2*pi, n, endpoint=False, retstep=True)
dt = h/4
c = .2 + sin(x-1)**2
v = exp(-100*(x-1)**2)
v_old = exp(-100*(x-.2*dt-1)**2)

# Time stepping; Leap-Frog formula
t_max = 8; t_plot = .35; n_plots = round(t_max/t_plot)
t = arange(0, t_max, step=dt)
data = zeros((len(x), len(t)))
for i, _ in enumerate(t):
    w = irfft(1j*rfftfreq(n, h/(2*pi))*rfft(v)).real
    v_new = v_old - 2*dt*c*w
    v_old = v; v = v_new
    data[:, i] = v

fig = plt.figure()
xx, tt = meshgrid(x, t[0:-1:n_plots])
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(xx, tt, data[:, 0:-1:n_plots].T, cstride=0, color="k", lw=.4)
ax.set_xlim(0, 2*pi); ax.set_ylim(0, t_max); ax.set_zlim(0, 4)
ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_zlabel("$u(x,t)$")
ax.view_init(30, -80); ax.grid(False)
plt.show()
