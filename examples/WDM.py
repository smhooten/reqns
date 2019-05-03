
import numpy as np
from math import pi
import matplotlib.pyplot as plt

import reqns
from reqns.physical_constants import h, hbar, c, q, eps0, m0, k


class WavelengthDivisionMultiplexer(object):
    def __init__(self, eff, *args):
        self.eff = eff

        self.reqns_objects = []
        for elem in args:
            assert isinstance(elem, reqns.rate_equations.RateEquations)
            self.reqns_objects.append(elem)

        self.num = len(reqns_objects)

 #   def find_optimal


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
A = 1.0/1.0e-6
C = 1e-28 * 1e-12

omega = np.linspace(0.5, 3.0, num = 1000) * q / hbar
DF_max = 1.0 * q
DF_dis = DF_max / 1000

mat = reqns.active_material.Bulk(omega, DF_max, DF_dis, Na, Nd, T, n, me, mh,
                                 M, Eg, A, C)

mat.build()


Fx = np.ones(omega.size)

antenna = reqns.LED.Antenna(omega,1,Fx)
sv_ratio = 2/(100e-9)
srv = 1.0e4 * 1e-2

led = reqns.LED.nanoLED(mat, antenna, srv, sv_ratio)
led.build(broaden_rspon=True)

t = [0.0, 1.0e-6]
max_step = 1.0e-11
I_func = lambda t: 0.001 + 1e-3*np.sin(2*pi*1e8*t)
V = 1.0e-18
N0 = [1.0e23]

rate = reqns.rate_equations.RateEquations(led, t, I_func, max_step=max_step,
                                          N0=N0, V=V)

nn = 1e18*1e6
I_guess = np.array([1e-5, 1e-3])

I_list, _ = rate.find_desired_Idc(nn, I_guess, option='nanoLED')

tt, Nt = rate.run('nanoLED')

print mat.correct_build
print led.correct_build
print mat.Nc / 1e6
print mat.Nv / 1e6

f1 = plt.figure()
ax1 = f1.add_subplot(111)
for i in range(mat.DF.size):
    ax1.plot(2*pi*c/omega*1e6, led.rspon_ant[:,i,0])

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

f7 = plt.figure()
ax7 = f7.add_subplot(111)
ax7.semilogy(tt, Nt)

plt.show()
