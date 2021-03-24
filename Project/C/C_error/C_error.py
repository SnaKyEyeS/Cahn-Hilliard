import numpy as np
from numpy.linalg import norm


# Load initial & reference solutions
# Reference solution is taken at t = 1e-3 w/ RK4 using dt = 1e-9
ref = np.loadtxt("rk4_128_1e-9.txt")

# RK4
rk4_6_16 = np.loadtxt("rk4_128_1e-6_16.txt");
rk4_6_32 = np.loadtxt("rk4_128_1e-6_32.txt");

e1 = norm(rk4_6_16 - ref, 2);
e2 = norm(rk4_6_32 - ref, 2);

n_rk4 = np.log(e1/e2)/np.log(2);

# ETDRK4
etdrk4_6_4 = np.loadtxt("etdrk4_128_1e-6_4.txt");
etdrk4_6_8 = np.loadtxt("etdrk4_128_1e-6_8.txt");
etdrk4_6_16 = np.loadtxt("etdrk4_128_1e-6_16.txt");
etdrk4_6_32 = np.loadtxt("etdrk4_128_1e-6_32.txt");
e1 = norm(etdrk4_6_4 - ref, 2)
e2 = norm(etdrk4_6_8 - ref, 2)

n_etdrk4 = np.log(e1/e2)/np.log(2)

# IMEX 2é
imex2_6_16  = np.loadtxt("imex2_128_1e-6_16.txt");
imex2_6_32  = np.loadtxt("imex2_128_1e-6_32.txt");
imex2_6_64  = np.loadtxt("imex2_128_1e-6_64.txt");
imex2_6_128 = np.loadtxt("imex2_128_1e-6_128.txt");

e1 = norm(imex2_6_16 - ref, 2);
e2 = norm(imex2_6_32 - ref, 2);
e3 = norm(imex2_6_64 - ref, 2);
e4 = norm(imex2_6_128 - ref, 2);

n_imex2 = np.log(e3/e4)/np.log(2);

# IMEX 4
imex4_6_16  = np.loadtxt("imex4_128_1e-6_16.txt");
imex4_6_32  = np.loadtxt("imex4_128_1e-6_32.txt");
imex4_6_64  = np.loadtxt("imex4_128_1e-6_64.txt");
imex4_6_128 = np.loadtxt("imex4_128_1e-6_128.txt");

e1 = norm(imex4_6_16 - ref, 2);
e2 = norm(imex4_6_32 - ref, 2);
e3 = norm(imex4_6_64 - ref, 2);
e4 = norm(imex4_6_128 - ref, 2);

n_imex4 = np.log(e3/e4)/np.log(2);

print("RK4: " + str(n_rk4) + ", IMEX 2: " + str(n_imex2) + ", IMEX 4: " + str(n_imex4) + ", ETDRK 4: " + str(n_etdrk4))
