import numpy as np
import matplotlib.pyplot as plt
from math import pi
plt.rcParams.update({'font.size':20})

filename = 'data_WDM_transfer8.npz'

files = np.load(filename)

deltaP_ant0 = files['deltaP_ant0']
deltaP_ant1 = files['deltaP_ant1']
deltaP_ant2 = files['deltaP_ant2']
deltaP_ant3 = files['deltaP_ant3']
w_master = files['w_master']
ns_master = files['ns_master']
I_master_ant0 = files['I_master'][:,0]
I_master_ant1 = files['I_master'][:,1]
I_master_ant2 = files['I_master'][:,2]
I_master_ant3 = files['I_master'][:,3]


print deltaP_ant0
print I_master_ant0

w_master = np.flipud(w_master)

f = plt.figure()
ax0 = f.add_subplot(221)
ax2 = f.add_subplot(222)
ax3 = f.add_subplot(223)
ax4 = f.add_subplot(224)

I1_ant0 = 0.5*I_master_ant0
I1_ant1 = 0.5*I_master_ant1
I1_ant2 = 0.5*I_master_ant2
I1_ant3 = 0.5*I_master_ant3


transfers_ant0 = np.zeros((ns_master.size, w_master.size))
transfers_ant1 = np.zeros((ns_master.size, w_master.size))
transfers_ant2 = np.zeros((ns_master.size, w_master.size))
transfers_ant3 = np.zeros((ns_master.size, w_master.size))

transfers_ant0_dc = np.zeros((ns_master.size))
transfers_ant1_dc = np.zeros((ns_master.size))
transfers_ant2_dc = np.zeros((ns_master.size))
transfers_ant3_dc = np.zeros((ns_master.size))

f3dB_ant0 = np.zeros((ns_master.size))
f3dB_ant1 = np.zeros((ns_master.size))
f3dB_ant2 = np.zeros((ns_master.size))
f3dB_ant3 = np.zeros((ns_master.size))
colors = np.linspace(0,1,num=ns_master.size)

V = 20e-9 * 100e-9 * 15e-9
q = 1.602e-19

for i in range(6,ns_master.size,1):
    print deltaP_ant0[i,0]
    transfers_ant0[i,:] = deltaP_ant0[i,:]/(2*I1_ant0[i])/(0.75)*V
    transfers_ant1[i,:] = deltaP_ant1[i,:]/(2*I1_ant1[i])/(0.75)*V
    transfers_ant2[i,:] = deltaP_ant2[i,:]/(2*I1_ant2[i])/(0.75)*V
    transfers_ant3[i,:] = deltaP_ant3[i,:]/(2*I1_ant3[i])/(0.75)*V
    
    transfers_ant0_dc[i] = 0.5*transfers_ant0[i,0]*V
    transfers_ant1_dc[i] = 0.5*transfers_ant1[i,0]*V
    transfers_ant2_dc[i] = 0.5*transfers_ant2[i,0]*V
    transfers_ant3_dc[i] = 0.5*transfers_ant3[i,0]*V
    
    transfers_ant0[i,:] = np.flipud(transfers_ant0[i,:])
    transfers_ant1[i,:] = np.flipud(transfers_ant1[i,:])
    transfers_ant2[i,:] = np.flipud(transfers_ant2[i,:])
    transfers_ant3[i,:] = np.flipud(transfers_ant3[i,:])
    
    f3dB_ant0[i] = np.interp(transfers_ant0_dc[i], np.squeeze(transfers_ant0[i,:]), w_master)
    f3dB_ant1[i] = np.interp(transfers_ant1_dc[i], np.squeeze(transfers_ant1[i,:]), w_master)
    f3dB_ant2[i] = np.interp(transfers_ant2_dc[i], np.squeeze(transfers_ant2[i,:]), w_master)
    f3dB_ant3[i] = np.interp(transfers_ant3_dc[i], np.squeeze(transfers_ant3[i,:]), w_master)
    
    f3dB_ant0[i] /= (2*pi)
    f3dB_ant1[i] /= (2*pi)
    f3dB_ant2[i] /= (2*pi)
    f3dB_ant3[i] /= (2*pi)
    #ax.loglog(transfers[i,:], w)
    #ax2.loglog(transfers_ant[i,:], w)

    ax0.loglog(w_master/(2*pi), transfers_ant0[i,:], color=(colors[i],0,1-colors[i]),linewidth=2)
    ax2.loglog(w_master/(2*pi), transfers_ant1[i,:], color=(colors[i],0,1-colors[i]),linewidth=2)
    ax3.loglog(w_master/(2*pi), transfers_ant2[i,:], color=(colors[i],0,1-colors[i]),linewidth=2)
    ax4.loglog(w_master/(2*pi), transfers_ant3[i,:], color=(colors[i],0,1-colors[i]),linewidth=2)
#ax.set_ylim([1e-2, 2])
#ax2.set_ylim([1e-2, 2])
#ax3.set_ylim([1e-2, 2])
#ax4.set_ylim([1e-2, 2])

ax0.set_xlabel('Frequency (Hz)')
ax2.set_xlabel('Frequency (Hz)')
ax3.set_xlabel('Frequency (Hz)')
ax4.set_xlabel('Frequency (Hz)')


f3 =plt.figure()
axxxx=f3.add_subplot(111)
for i in range(6,ns_master.size,1):
    ratio1 = 0.4*0.9*(transfers_ant1[i,:]+transfers_ant2[i,:]+transfers_ant3[i,:])
    ratio2 = 0.6*transfers_ant0[i,:]
    ratio = ratio1/ratio2
    line=axxxx.semilogx(w_master/(2*pi), ratio,linewidth=2)
    axxxx.set_xlim([w_master[-1]/(2*pi), 1e12])
    axxxx.set_ylim([1, 3.5])
    axxxx.set_xlabel('Modulation Frequency (Hz)')
    axxxx.set_ylabel(r'$\Delta P_{WDM}/\Delta P_{ref}$')
    print ns_master[i]/1e6

f2 = plt.figure()
axx = f2.add_subplot(111)
axx.semilogx(ns_master/1e6, f3dB_ant0)
axx.semilogx(ns_master/1e6, f3dB_ant1)
axx.semilogx(ns_master/1e6, f3dB_ant2)
axx.semilogx(ns_master/1e6, f3dB_ant3)
plt.show()
