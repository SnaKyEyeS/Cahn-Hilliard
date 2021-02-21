from numpy import arange, meshgrid, pi, exp
import matplotlib.pyplot as plt


n = 70
theta = -pi + 2*pi/n * arange(.5, n+.5)
c = 11      # Center of circle of integration
r = 16      # Radius of circle of integration
x = arange(-3.5, 4.1, step=.1)
y = arange(-2.5, 2.6, step=.1)
xx, yy = meshgrid(x, y)
zz = xx + 1j*yy
gam_inv = 0*zz

for i in range(n):
    t = c + r*exp(1j*theta[i])
    gam_inv = gam_inv + exp(t)*t**zz*(t - c)
gam = n / gam_inv

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(xx, yy, abs(gam), cmap="viridis", edgecolor="none")
ax.set_xlabel("Re(z)")
ax.set_ylabel("Im(z)")
ax.set_xlim(-3.5, 4)
ax.set_ylim(-2.5, 2.5)
ax.set_zlim(0, 6)

plt.show()
