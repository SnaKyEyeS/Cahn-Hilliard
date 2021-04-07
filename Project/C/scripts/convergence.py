from os import walk

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt


# Load reference solutions
# Reference solution is taken at t = 1e-3 w/ RK4 using dt = 1e-9
ref = np.loadtxt("rk4_128_1e-9.txt")

# Load data & compute errors
etdrk4 = {"dt": list(), "L2": list(), "Linf": list()}
imex2 = {"dt": list(), "L2": list(), "Linf": list()}
imex4 = {"dt": list(), "L2": list(), "Linf": list()}
rk4 = {"dt": list(), "L2": list(), "Linf": list()}

_, _, filenames = next(walk("./data"))
for filename in filenames:
    data = np.loadtxt("data/" + filename)

    scheme = filename.split("_")[0]
    dt = float(filename.split("_")[2].split(".txt")[0])

    if scheme == "etdrk4":
        etdrk4["dt"].append(dt)
        etdrk4["L2"].append(norm(data - ref) / norm(ref))
        etdrk4["Linf"].append(norm(data - ref, np.inf) / norm(ref, np.inf))

    if scheme == "imex2":
        imex2["dt"].append(dt)
        imex2["L2"].append(norm(data - ref) / norm(ref))
        imex2["Linf"].append(norm(data - ref, np.inf) / norm(ref, np.inf))

    if scheme == "imex4":
        imex4["dt"].append(dt)
        imex4["L2"].append(norm(data - ref) / norm(ref))
        imex4["Linf"].append(norm(data - ref, np.inf) / norm(ref, np.inf))

    if scheme == "rk4":
        rk4["dt"].append(dt)
        rk4["L2"].append(norm(data - ref) / norm(ref))
        rk4["Linf"].append(norm(data - ref, np.inf) / norm(ref, np.inf))


# Sort results
etdrk4["L2"] = [e for _, e in sorted(zip(etdrk4["dt"], etdrk4["L2"]))]
etdrk4["Linf"] = [e for _, e in sorted(zip(etdrk4["dt"], etdrk4["Linf"]))]
etdrk4["dt"].sort()

imex2["L2"] = [e for _, e in sorted(zip(imex2["dt"], imex2["L2"]))]
imex2["Linf"] = [e for _, e in sorted(zip(imex2["dt"], imex2["Linf"]))]
imex2["dt"].sort()

imex4["L2"] = [e for _, e in sorted(zip(imex4["dt"], imex4["L2"]))]
imex4["Linf"] = [e for _, e in sorted(zip(imex4["dt"], imex4["Linf"]))]
imex4["dt"].sort()

rk4["L2"] = [e for _, e in sorted(zip(rk4["dt"], rk4["L2"]))]
rk4["Linf"] = [e for _, e in sorted(zip(rk4["dt"], rk4["Linf"]))]
rk4["dt"].sort()

# Plot results
plt.rcParams.update({"text.usetex": True})

plt.loglog(etdrk4["dt"], etdrk4["L2"], "--v", label="ETDRK4")
plt.loglog(imex2["dt"], imex2["L2"], "--o", label="IMEX2")
plt.loglog(imex4["dt"], imex4["L2"], "--D", label="IMEX4")
plt.loglog(rk4["dt"], rk4["L2"], "--s", label="RK4")

plt.loglog(np.array([5e-7, 1.8e-8]), 2e10*np.array([5e-7, 1.8e-8])**2, label="$\\mathcal{O}(\\Delta t^2)$")
plt.loglog(np.array([1e-7, 1.2e-8]), 2e20*np.array([1e-7, 1.2e-8])**4, label="$\\mathcal{O}(\\Delta t^4)$")

plt.xlabel("Time step $\\Delta t$")
plt.ylabel("L$_2$ norm relative error at $t = 10^{-3}$")
plt.grid(True, which="both", ls="--")
plt.gca().invert_xaxis()
plt.legend()
plt.show()
# plt.savefig("convergence.pdf")
