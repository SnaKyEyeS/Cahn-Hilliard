import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

N = [4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024]
python = [0.15956759452819824, 0.18604469299316406, 0.23384451866149902, 0.3759739398956299, 0.8854141235351562, 2.9699997901916504, 6.59096622467041, 11.677207469940186, 27.478625535964966, 50.5753231048584, 116.0897011756897, 210.91827845573425]
MKL = [0.001040, 0.004100, 0.011041, 0.020420, 0.088646, 0.381456, 1.023830, 1.879621, 5.218947, 10.575320, 30.000266, 57.385178]
FFTW = [0.000512, 0.003270, 0.005726, 0.026156, 0.128126, 0.636778, 1.582049, 2.746788, 6.897447, 14.302295, 35.522980, 67.871003]


fig, ax = plt.subplots()
mpl.rcParams.update({"text.usetex": True})

python_plot = ax.plot(N, python, "--v", label='python')
FFTW_plot = ax.plot(N, FFTW, "--o", label='FFTW')
MKL_plot = ax.plot(N, MKL, "--D", label='MKL')

# reference en N log N
x = np.linspace(500, 1024)
ref_fact = (x**2*np.log(x))/(768**2*np.log(768))
ref = 200 * (ref_fact)

ref_plot = ax.plot(x, ref, 'k--', label='$\\mathcal{O}(n^2\\log(n))$')

ax.set_yscale("log")
ax.set_xscale("log", base=2)
plt.grid(True, which="both", ls="--")
plt.xlabel("n")
plt.ylabel("Execution time [s]")
plt.legend()
plt.show()
