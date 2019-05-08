import numpy as np
from math import pi
import matplotlib.pyplot as plt

import reqns
from reqns.physical_constants import h, hbar, c, q, eps0, m0, k

import matplotlib.pyplot as plt
n = 3.5
Egw = 0.755*q
Ep = 25.7 * q
Nd = 0.0
Na = 0.0
Lz = 5.0e-9
M = (m0/6)*Ep
mw = 0.041*m0
mw_lh = 0.0503*m0
mw_hh = 0.46*m0

A = 1.0/1.0e-6
C = 1e-28 * 1e-12

mb = 0.065*m0
mb_lh = 0.087*m0
mb_hh = 0.46*m0

Egb = 1.03*q

delEc = 0.4

omega = np.linspace(0.5, 2.5, num=2000)*q/hbar
DF_max = 1.5*q
DF_dis = 1.5*q/200
T = 77.0

mat = reqns.active_material.QuantumWell(omega, DF_max, DF_dis, Na, Nd, T, n, M, Egw, Lz, mw, mw_lh,
                 mw_hh, A, C, Egb=Egb, mb=mb, mb_lh=mb_lh, mb_hh=mb_hh, delEc=delEc)

#QW = QuantumWell(omega, DF_max, DF_dis, Na, Nd, T, n, M, Egw, Lz, mw, mw_lh, mw_hh)

mat.build(broaden_gain = False)



Fx = 10*np.ones(omega.size)

antenna = reqns.LED.Antenna(omega, Fx)
sv_ratio = 2/(100e-9)
srv = 1.0e4 * 1e-2

led = reqns.LED.nanoLED(mat, antenna, srv, sv_ratio)
led.build(broaden_rspon = True)


print mat.correct_build
print led.correct_build
print mat.Nc / 1e6

f1 = plt.figure()
ax1 = f1.add_subplot(111)
for i in range(mat.DF.size):
    ax1.plot(2*pi*c/omega*1e6, led.rspon_ant_broadened[:,i,0])

f5 = plt.figure()
ax5 = f5.add_subplot(111)
for i in range(mat.DF.size):
    ax5.plot(2*pi*c/omega*1e6, led.rspon_ant[:,i,1])

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.semilogy(mat.DF / q, mat.N/1e6)
ax2.semilogy(mat.DF / q, mat.P/1e6)

f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.loglog(mat.N / 1e6, mat.N / led.Rspon_ant)

f4 = plt.figure()
ax4 = f4.add_subplot(111)
ax4.semilogx(mat.N / 1e6, led.Rspon/(led.Rspon_ant + led.Rnr_ant))

plt.show()
