import reqns
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from math import pi

q = 1.602e-19
c = 299792458
h = 6.63e-34
hbar = 1.055e-34
m0 = 9.11e-31

data = scipy.io.loadmat('dipole_antenna_enhancement_InP_Gap.mat')
d = data['d']
enhancement = data['enhancement']
ant_efficiency_yesASE = data['efficiency_yesASE']
ant_efficiency_noASE = data['efficiency_noASE']
print(enhancement)
lorentzian = lambda w, w0, Q: (4*Q**2*((w-w0)/w0)**2+1)**(-1)

n = 3.5
Eg = 0.8*q
Ep = 25.7 * q
Nd = 0.0

Nas = np.logspace(0,21,num=1000)*1e6
M = (m0/6)*Ep

#CHECK THESE
me = 0.05*m0
mh = 0.4*m0


#DF_max = 1.1*q
#DF_dis = 1.1*q/500
DF_max = Eg
DF_dis = Eg/500
T=300.0

A = 1e8
C = 1e-28

omega = np.linspace(0.1, 3.0, num=1000)*q/hbar
w0 = 2*pi*c/1550e-9
Q = 4.5

sv_ratio = 2.0/d
srv = 1e4*1e-2

ds = np.linspace(1,50,num=5000)*1e-9
eta_IQEs = np.zeros((Nas.shape[0], ds.shape[0]))
rad_lifetimes = np.zeros((Nas.shape[0], ds.shape[0]))

for i in range(Nas.shape[0]):
    for j in range(ds.shape[0]):
        Na = Nas[i]
        bulk = reqns.active_material.Bulk(omega, DF_max, DF_dis, Na, Nd, T, n, me, mh, M, Eg, A, C)
        bulk.build()

        Fx = lorentzian(omega, w0, Q) * enhancement[j]
        antenna = reqns.LED.Antenna(omega, 1.0, Fx)
        
        nanoLED = reqns.LED.nanoLED(bulk, antenna, srv, sv_ratio)
        nanoLED.build()
        
        rad_lifetime = bulk._rad_lifetime_ant[-2]
        nr_lifetime = bulk._nr_lifetime_ant[-2]
        eta_IQE = rad_lifetime/(rad_lifetime+nr_lifetime)

        rad_lifetimes[i,j] = rad_lifetime
        eta_IQEs[i,j] = eta_IQE
        del bulk

f1 = plt.figure()
f2 = plt.figure()
ax1 = f.add_subplot(111)
ax2 = f2.add_subplot(111)

ax1.imshow(eta_IQEs, extent=[np.log(Nas[0]), np.log(Nas[-1]), 1e9*ds[0], 1e9*ds[1]], aspect = 1, cmap = 'hot')

ax2.imshow(rad_lifetimes, extent=[np.log(Nas[0]), np.log(Nas[-1]), 1e9*ds[0], 1e9*ds[1]], aspect = 1, cmap = 'hot')

plt.show()
