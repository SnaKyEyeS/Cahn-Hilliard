from numpy import cos, linspace, meshgrid, exp, pi
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


# Grid & initial data
n = 24
x = cos(pi*linspace(0, 1, n))
dt = 6/n**2
xx, yy = meshgrid(x, x)
vv = exp(-40*((xx-.4)**2 + yy**2))
vv_old = vv
