from numpy import cos, meshgrid, exp, pi, remainder, zeros, arange, flipud, linspace, concatenate
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


# Grid & initial data
n = 24; x = cos(pi*linspace(0, 1, n+1)); y = x
dt = 6/n**2; plotgap = round(1./3./dt); dt = 1/3/plotgap
xx, yy = meshgrid(x, y)
vv = exp(-40*((xx-.4)**2 + yy**2)); vv_old = vv

# Time stepping by Leap-Frog formula
fig = plt.figure()
for k in range(0, 3*plotgap+1):
    if remainder(k+.5, plotgap) < 1:    # Plot at multiples of t = 1/3
        interp = arange(-1, 1+1/16, 1/16)
        xxx, yyy = meshgrid(interp, interp)
        vvv = interp2d(x, y, vv, kind="cubic")(interp, interp)
        ax = fig.add_subplot(2, 2, int(k/plotgap + 1), projection="3d")
        ax.plot_surface(xxx, yyy, vvv, cmap="jet", edgecolor="black")
        ax.set_xlim3d([-1, 1]); ax.set_ylim3d([-1, 1]), ax.set_zlim3d([-.15, 1])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(f"t = {k*dt:.4f}")
        ax.view_init(18, 240)

    uxx = zeros((n+1, n+1)); uyy = zeros((n+1, n+1))
    ii = arange(1, n)

    for i in range(1, n):               # 2nd deriv w.r.t. x in each row
        v = vv[i, :]; V = concatenate((v, flipud(v[ii])))
        U = fft(V).real
        W1 = ifft(1j*fftfreq(2*n, .5/n) * U).real      # diff   w.r.t. theta
        W2 = ifft(-fftfreq(2*n, .5/n)**2 * U).real     # diff^2 w.r.t. theta
        uxx[i, ii] = W2[ii]/(1 - x[ii]**2) - x[ii]*W1[ii]/(1 - x[ii]**2)**1.5

    for j in range(1, n):               # 2nd deriv wrt y in each row
        v = vv[:, j]; V = concatenate((v, flipud(v[ii])))
        U = fft(V).real
        W1 = ifft(1j*fftfreq(2*n, .5/n) * U).real      # diff   w.r.t. theta
        W2 = ifft(-fftfreq(2*n, .5/n)**2 * U).real     # diff^2 w.r.t. theta
        uyy[ii, j] = W2[ii]/(1 - y[ii]**2) - y[ii]*W1[ii]/(1 - y[ii]**2)**1.5

    vv_new = 2*vv - vv_old + dt**2 * (uxx + uyy)
    vv_old = vv; vv = vv_new
plt.show()
