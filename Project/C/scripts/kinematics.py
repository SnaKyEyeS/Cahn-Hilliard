import itertools
from os import walk
from tqdm import tqdm

import numpy as np
from scipy.fft import fft2, fftfreq
from scipy.optimize import curve_fit as fit
from scipy.interpolate import CubicSpline as spline

from matplotlib import pyplot as plt


# Wavenumber
n = 1024
x, h = np.linspace(0, n, n, endpoint=False, retstep=True)
k_x, k_y = np.meshgrid(fftfreq(n, h/(2*np.pi)), fftfreq(n, h/(2*np.pi)))
k = np.sqrt(k_x**2 + k_y**2)
k_radius, dr = np.linspace(0, 0.5, 25, endpoint=True, retstep=True)

# Load data
k1 = list()
s = list()
t = list()
n_k_max = 1000

mobility = "variable"

_, _, filenames = next(walk(f"./data/{mobility}"))
for filename in tqdm(filenames):

    # Load data
    data = np.loadtxt(f"data/{mobility}/" + filename).reshape(n, n)
    t.append(int(filename.split("_")[3].split(".txt")[0]))

    # Compute S
    c_avg = np.mean(data, axis=(0, 1))
    c_hat = fft2(data - c_avg)
    Skk = (c_hat * np.conjugate(c_hat)).real

    # Normalize S
    c_int = np.sum((data - c_avg)**2, axis=(0, 1))
    s_int = np.sum(Skk, axis=(0, 1))
    Skk *= c_int / s_int

    # Compute circular integral
    S = np.zeros(k_radius.shape)
    for i, k_val in enumerate(k_radius):
        row, col = np.where(np.abs(k - k_val) < dr/2)
        S[i] = np.mean(Skk[row, col])
    s.append(S)

    # Compute k1
    k1.append(np.sum(k_radius*S) / np.sum(S))


# Sort w.r.t. t
s = [e for _, e in sorted(zip(t, s))]
k1 = np.array([e for _, e in sorted(zip(t, k1))])
t.sort()

# Plot structure function S(k,t)
plt.rcParams.update({"text.usetex": True})
ax = plt.gca()
for mat, t_ in zip(s, t):
    color = next(ax._get_lines.prop_cycler)['color']
    k_interp = np.linspace(k_radius[0], k_radius[-1], 1000, endpoint=True)
    s_interp = spline(k_radius, mat, bc_type="clamped")(k_interp)
    plt.plot(k_interp, s_interp, label=f"t = {t_}", color=color)
    # plt.plot(k_radius, mat, "o", color=color)
plt.xlabel("k")
plt.ylabel("S(k,t)")
plt.xlim([k_radius[0], k_radius[-1]])
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()
# plt.savefig(f"structure_{mobility}.pdf")

# Plot F, the scaling function
plt.clf()
idx = -6
marker = itertools.cycle(('+', '.', 'o', '*', '<', 'd', 'x', '1'))
for s_, k1_, t_ in zip(s[idx:], k1[idx:], t[idx:]):
    plt.plot(k_radius/k1_, s_*k1_**2, linestyle="", marker=next(marker), label=f"t = {t_}")
plt.xlabel("$k/k_1$")
plt.ylabel("$k_1^2s(k,t)$")
plt.gca().set_xlim(left=0, right=3)
plt.gca().set_ylim(bottom=0)
plt.legend()
plt.show()
# plt.savefig(f"scalefunc_{mobility}.pdf")

# Compute power growth law
f = lambda x, a, b, m: a + b*x**m
idx = 0
t = np.array(t)
popt, pcov = fit(f, t[idx:], 1/k1[idx:], bounds=(0, [np.inf, np.inf, .5]), maxfev=10000)
print(f"Power growth law exponent m = {popt[-1]}")

# Plot L(t)
plt.clf()
ref = t[idx:]**(1./4.)
ref /= ref[-1] * k1[-1]

plt.loglog(t[idx:], ref, label="$\\mathcal{O}\\left(\\sqrt[4]{t}\\right)$")
plt.loglog(t[idx:], 1/k1[idx:], "o", label="Simulation data")
plt.xlabel("t")
plt.ylabel("L(t)")
plt.legend()
plt.show()
# plt.savefig(f"length_{mobility}.pdf")
