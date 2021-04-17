#ExCM3_wave Quiriny Antoine
from numpy import *
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift
from scipy.integrate import solve_ivp as RK45

def deriv_x(u):
    N = len(u)
    f_hat = fftshift(fft(u))
    k = arange(-N/2, N/2)

    f_hat_dot = 1j*k*f_hat
    f_dot = ifft(fftshift(f_hat_dot)).real
    return f_dot

def deriv_t(t, u):
    N = 100
    x = linspace(0,2*pi, N)
    c = (1/5) + (sin(x-1))**2
    return -c*deriv_x(u)


#time discretisation
t_0 = 0
t_end = 5
t = linspace(t_0, t_end, 501)

#space discretisation
N = 100
x = linspace(0,2*pi, N)
h = 2*pi/N

#init space domain
x_init = exp(-100*(x-1)**2) #init of the book

#runge kutta
sol = RK45(deriv_t, [t_0, t_end], x_init, t_eval = t)

# Plot results
x_grid, t_grid = meshgrid(x, sol.t)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(x_grid, t_grid, sol.y.T, cmap="viridis")

ax.set_xlim(0, x[-1])
ax.set_ylim(0, t_end)
ax.set_zlim(0, 4)

ax.set_xlabel("X axis")
ax.set_ylabel("Time axis")

ax.view_init(30, 100)

plt.show()
