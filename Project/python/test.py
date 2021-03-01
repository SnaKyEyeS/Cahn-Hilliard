import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import rfft2, irfft2, rfftfreq, fftfreq


n = 1000
discr, h = np.linspace(0, 2*np.pi, n, retstep=True, endpoint=False)
x, y = np.meshgrid(discr, discr)

# Wave number
k_x, k_y = np.meshgrid(rfftfreq(n, h/(2*np.pi)), fftfreq(n, h/(2*np.pi)))
k = -(k_x**2 + k_y**2)

# Differentiate
f = np.cos(x)*np.sin(y)

df = irfft2(k*rfft2(f**3 - f - irfft2(k*rfft2(f))))

# df = irfft2(k*rfft2(f))
# df_exact = -2*np.cos(x)*np.sin(y)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, df, cmap="coolwarm")
plt.show()
