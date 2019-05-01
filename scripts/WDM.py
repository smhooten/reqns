import numpy as np
from math import pi
import matplotlib.pyplot as plt

import reqns
from reqns.physical_constants import h, hbar, c, q, eps0, m0, k

# Bulk InGaAs
T = 300.0
Na = 0.0
Nd = 0.0

n = 3.5
Eg = 0.75 * q
me = 0.041 * m0
mh = 0.465 * m0

Ep = 25.7 * q
M = (m0/6) * Ep


omega = np.linspace(0.5, 3.0, num = 2000) * q / hbar
DF_max = 1.0 * q
DF_dis = DF_max / 1000

mat = reqns.active_material.Bulk(omega, DF_max, DF_dis, Na, Nd, T, n, me, mh,
                                 M, Eg)


print active_mat.correct_build
print active_mat.Nc / 1e6
print active_mat.Nv / 1e6

f1 = plt.figure()
ax1 = f1.add_subplot(111)
for i in range(active_mat.DF.size):
    ax1.plot(2*pi*c/omega*1e6, active_mat.gain[:,i])

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.semilogy(active_mat.DF / q, active_mat.N/1e6)
ax2.semilogy(active_mat.DF / q, active_mat.P/1e6)

plt.show()
