from numpy import linspace, pi, sin, arange, zeros, meshgrid, array, exp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


# Grid & initial data
n = 128
x, h = linspace(0, 2*pi, n, endpoint=False, retstep=True)
dt = h/4
c = .2 + sin(x-1)**2
v = exp(-100*(x-1)**2)
v_old = exp(-100*(x-.2*dt-1)**2)

# Time stepping; Leap-Frog formula
t_max = 8
t = arange(0, t_max, step=dt)
data = zeros((len(x), len(t)))
data[:, 0] = v
for i, _ in enumerate(t):
    v_hat = fft(v)
    w_hat = 1j*array([*arange(n/2), 0, *arange(-n/2+1, 0)])*v_hat
    w = ifft(w_hat).real
    v_new = v_old - 2*dt*c*w
    v_old = v
    v = v_new
    data[:, i] = v

# Plot
xx, tt = meshgrid(x, t[0:-1:20])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_wireframe(xx, tt, data[:, 0:-1:20].T, cstride=0)
ax.set_xlim(0, x[-1])
ax.set_ylim(0, t[-1])
ax.set_zlim(0, 4)
ax.set_xlabel("X axis")
ax.set_ylabel("Time axis")
ax.view_init(30, -80)
ax.grid(False)
plt.show()
