import copy

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import reqns
from reqns.physical_constants import h, hbar, c, q, eps0, m0, k

from reqns.parallel_toolkit import parallel_partition, create_master_array, COMM, SIZE, RANK


plt.rcParams.update({'font.size':24, 'font.weight':'bold', 'font.family':'arial',
                     'axes.linewidth':2, 'lines.linewidth':3})

#def sawtooth(t, I0, dI, w):
#    


class WavelengthDivisionMultiplexer(object):
    def __init__(self, eff, *args):
        self.eff = eff
        self.reqns_objects = []
        for elem in args:
            assert isinstance(elem, reqns.rate_equations.RateEquations)
            self.reqns_objects.append(elem)

        self.omega = self.reqns_objects[0]._LED.omega
        self.num = len(self.reqns_objects)

    def find_optimal_window(self, w0s):
        powers = []
        w0inds = []
        inds = []
        for i in range(self.num):
            reqn = self.reqns_objects[i]
            powers.append(reqn._LED._antenna.Fx * \
                          reqn._LED._antenna.efficiency)
            w0inds.append(np.argmin(np.abs(self.omega-w0s[i])))

        for i in range(self.num-1):
            ind = np.argmin(np.abs(powers[i][w0inds[i]:w0inds[i+1]]-powers[i+1][w0inds[i]:w0inds[i+1]]))
            inds.append(w0inds[i]+ind)


        print inds
        windows = []
        windows_inds = []

        windows.append([self.omega[0],self.omega[inds[0]]])
        windows_inds.append([0, inds[0]])
        for i in range(self.num-2):
            windows.append([self.omega[inds[i]], self.omega[inds[i+1]]])
            windows_inds.append([inds[i], inds[i+1]])

        windows.append([self.omega[inds[-1]], self.omega[-1]])
        windows_inds.append([inds[-1], self.omega.size-1])

        window = []
        for i in range(self.num):
            window.append(np.zeros(omega.size))
            window[i][windows_inds[i][0]:windows_inds[i][1]] = 1.0

        self.windows = window
        self.windows_inds = windows_inds

    def find_collected_power(self, N):
        fractions = np.zeros(self.num)
        for i in range(self.num):
            inds = self.windows_inds[i]
            j = np.argmin(np.abs(N-self.reqns_objects[i]._LED._active_mat.N))
            reqn = self.reqns_objects[i]
            #f = plt.figure()
            #ax = f.add_subplot(111)
            #ax.loglog(reqn._LED._active_mat.N, reqn._LED.hvRspon_ant)
            #ax.loglog(reqn._LED._active_mat.N, reqn._LED.hvRspon)
            #f2 = plt.figure()
            #ax2 = f2.add_subplot(111)
            #ax2.loglog(reqn._LED._active_mat.N, reqn._LED.Rspon_ant)
            #ax2.loglog(reqn._LED._active_mat.N, reqn._LED.Rspon)
            #f3 = plt.figure()
            #ax3 = f3.add_subplot(111)
            #ax3.plot(reqn._LED._active_mat.omega, reqn._LED.rspon_ant[:,-1,0])
            #ax3.plot(reqn._LED._active_mat.omega, reqn._LED.rspon[:,-1,0])
            #plt.show()

            eff = reqn._LED._antenna.efficiency
            rspon_ant = np.sum(reqn._LED.rspon_ant[:,j,:], axis=1)
            av1 = np.trapz(hbar*self.omega*rspon_ant, x=hbar*self.omega)
            av2 = np.trapz(hbar*self.omega*rspon_ant*self.windows[i]*eff, x=hbar*self.omega)
            integral2 = self.eff*av2/av1
            integral = self.eff*av2/reqn._LED.hvRspon_ant[j]
            print integral2, integral
            fractions[i] = integral

        self.fractions = fractions

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
A = 1.0/1.0e-8
C = 1e-28 * 1e-12

omega = np.linspace(0.5, 3.0, num = 2000) * q / hbar
DF_max = 1.5 * q
DF_dis = DF_max / 1500

mat = reqns.active_material.Bulk(omega, DF_max, DF_dis, Na, Nd, T, n, me, mh,
                                 M, Eg, A, C)

mat.build()


efficiency0 = 0.6*np.ones(omega.size)
efficiency1 = 0.4*np.ones(omega.size)
efficiency2 = 0.4*np.ones(omega.size)
efficiency3 = 0.4*np.ones(omega.size)

lam1 = 2*pi*c/(1.55e-6)
lam2 = 2*pi*c/(1.4e-6)
lam3 = 2*pi*c/(1.25e-6)
Fx0 = reqns.LED.purcell_enhancement(omega, lam2, 10.0, 1.0/500)
Fx1 = reqns.LED.purcell_enhancement(omega, lam1, 30.0, 1.0/500)
Fx2 = reqns.LED.purcell_enhancement(omega, lam2, 30.0, 1.0/500)
Fx3 = reqns.LED.purcell_enhancement(omega, lam3, 30.0, 1.0/500)
#f = plt.figure()
#f2 = plt.figure()
#ax = f.add_subplot(111)
#ax2 = f2.add_subplot(111)
#ax.plot(2*pi*c/omega*1e6, Fx2,'-r')
#ax2.plot(2*pi*c/omega*1e6, Fx3, '-b')
##ax.plot(omega*hbar/q, Fx3)
#plt.show()

antenna0 = reqns.LED.Antenna(omega,efficiency0,Fx0)
antenna1 = reqns.LED.Antenna(omega,efficiency1,Fx1)
antenna2 = reqns.LED.Antenna(omega,efficiency2,Fx2)
antenna3 = reqns.LED.Antenna(omega,efficiency3,Fx3)
sv_ratio = 2/(1000e-9)
srv = 1.0e2 * 1e-2

led1 = reqns.LED.nanoLED(mat, antenna1, srv, sv_ratio)
led0 = copy.deepcopy(led1)
led0._antenna = antenna0
led2 = copy.deepcopy(led1)
led2._antenna = antenna2
led3 = copy.deepcopy(led1)
led3._antenna = antenna3

led0.build(broaden_rspon=False)
led1.build(broaden_rspon=False)
led2.build(broaden_rspon=False)
led3.build(broaden_rspon=False)


#f = plt.figure()
#ax = f.add_subplot(111)
#f2 = plt.figure()
#ax2 = f2.add_subplot(111)
#f3 = plt.figure()
#ax3 = f3.add_subplot(111)
#ax.plot(2*pi*c/omega*1e6, led2.rspon_ant[:,-1,0]/np.max(led2.rspon[:,-1,0]),'-r')
#ax2.plot(2*pi*c/omega*1e6,led3.rspon_ant[:,-1,0]/np.max(led3.rspon[:,-1,0]),'-b')
#ax.set_ylim([-0.5,80.5])
#ax2.set_ylim([-0.5,80.5])
#
#wind1 = np.ones(omega.size)
#wind1[316:] = 0.0
#
#wind2 = np.ones(omega.size)
#wind2[:316] = 0.0
#
#yy = wind1*led2.rspon_ant[:,-1,0]/np.max(led2.rspon[:,-1,0]) + \
#        wind2*led3.rspon_ant[:,-1,0]/np.max(led3.rspon[:,-1,0])
#
#ax3.plot(2*pi*c/omega*1e6, yy ,'-k')
#ax3.set_ylim([-0.5,80.5])
#plt.show()

#f0 = plt.figure()
#ax0 = f0.add_subplot(111)
#f1 = plt.figure()
#ax1 = f1.add_subplot(111)
#f2 = plt.figure()
#ax2 = f2.add_subplot(111)
#f3 = plt.figure()
#ax3 = f3.add_subplot(111)
#
#ax0.plot(2*pi*c/omega*1e6, Fx0,'-k')
#ax1.plot(2*pi*c/omega*1e6, Fx1,'-b')
#ax2.plot(2*pi*c/omega*1e6, Fx2,'-g')
#ax3.plot(2*pi*c/omega*1e6, Fx3,'-r')
#ax0.set_ylim([0, 2500])
#ax1.set_ylim([0, 2500])
#ax2.set_ylim([0, 2500])
#ax3.set_ylim([0, 2500])
#ax0.set_xlim([1.0, 1.8])
#ax1.set_xlim([1.0, 1.8])
#ax2.set_xlim([1.0, 1.8])
#ax3.set_xlim([1.0, 1.8])
##ax0.plot(2*pi*c/omega*1e6, led0.rspon[:,-1,0]/np.max(led0.rspon[:,-1,0])*1000,'--k')
#
#plt.show()


t = [0.0, 1.0e-7]
max_step = 1.0e-11
I_func = lambda t: 0.001 + 1e-3*np.sin(2*pi*1e8*t)
V = 20e-9 * 100e-9 * 15e-9
N0 = [1.0e23]

rate0 = reqns.rate_equations.RateEquations(led0, t, I_func, max_step=max_step,
                                          N0=N0, V=V)
rate1 = reqns.rate_equations.RateEquations(led1, t, I_func, max_step=max_step,
                                          N0=N0, V=V)
rate2 = reqns.rate_equations.RateEquations(led2, t, I_func, max_step=max_step,
                                          N0=N0, V=V)
rate3 = reqns.rate_equations.RateEquations(led3, t, I_func, max_step=max_step,
                                          N0=N0, V=V)


wdm = WavelengthDivisionMultiplexer(0.9,rate1, rate2, rate3)
w0s = [2*pi*c/(1.55e-6), 2*pi*c/(1.4e-6), 2*pi*c/(1.25e-6)]
wdm.find_optimal_window(w0s)
wdm.find_collected_power(5.0e18*1e6)

#print np.array(wdm.windows)*hbar/q
#print wdm.windows_inds
ns_master = np.linspace(1.0e17,1.0e19,num=21)*1e6
#nn = 5.0e24
I_guess = np.array([1e-9, 1.0e-3])

#I_list, _ = rate0.find_desired_Idc(nn, I_guess, option='nanoLED')

#rate0.I_func = lambda t: I_list[0]+I_list[0]*np.sin(2*pi*1e9*t)

#tt, Nt = rate0.run('nanoLED')
#f7 = plt.figure()
#ax7 = f7.add_subplot(111)
#ax7.plot(tt*1e9, Nt/1e6)
#f8 = plt.figure()
#ax8 = f8.add_subplot(111)
#ax8.plot(tt*1e9, rate0.I_func(tt))

#plt.show()
num_sims = int(ns_master.size)
partition = parallel_partition(num_sims)

ns_master_curr = ns_master[partition]

# Find corresponding DC currents
I_bulk = np.zeros(len(partition))
I_ant = np.zeros(len(partition))
for i in range(len(partition)):
    nss = ns_master_curr[i]
    I_ant0, _ = rate0.find_desired_Idc(nss, I_guess, option = 'nanoLED')
    I_ant1, _ = rate1.find_desired_Idc(nss, I_guess, option = 'nanoLED')
    I_ant2, _ = rate2.find_desired_Idc(nss, I_guess, option = 'nanoLED')
    I_ant3, _ = rate3.find_desired_Idc(nss, I_guess, option = 'nanoLED')

print('Node {0} completed current search'.format(RANK))

I_ant0 = COMM.gather(I_ant0, root=0)
I_ant1 = COMM.gather(I_ant1, root=0)
I_ant2 = COMM.gather(I_ant2, root=0)
I_ant3 = COMM.gather(I_ant3, root=0)
I_master = create_master_array(num_sims, I_ant0, I_ant1, I_ant2, I_ant3)
I_master = COMM.bcast(I_master, root=0)

w_master = 2*pi*np.logspace(7, 12, num=31)

num_sims = int(w_master.size * ns_master.size)
partition = parallel_partition(num_sims)


simulations = [(II,ww) for II in range(ns_master.size) for ww in w_master]

simulations_curr = [simulations[i] for i in partition]


deltaP_ant0 = np.zeros(len(simulations_curr))
deltaP_ant1 = np.zeros(len(simulations_curr))
deltaP_ant2 = np.zeros(len(simulations_curr))
deltaP_ant3 = np.zeros(len(simulations_curr))


for i in range(len(simulations_curr)):

    print('Node {0} is performing simulation {1} of {2}'.format(RANK, i,
                                                                len(simulations_curr)))
    w = simulations_curr[i][1]
    index = simulations_curr[i][0]

    I_ant0 = I_master[index,0]
    I_ant1 = I_master[index,1]
    I_ant2 = I_master[index,2]
    I_ant3 = I_master[index,3]

    I1_ant0 = 0.5*I_ant0
    I1_ant1 = 0.5*I_ant1
    I1_ant2 = 0.5*I_ant2
    I1_ant3 = 0.5*I_ant3


    time = [0.0, 1e-8]
    time[1] = 10*2*pi/w if 10*2*pi/w > 1.0e-8 else 1.0e-8

    I_func = lambda t: I_ant0 + I1_ant0*np.sin(w*t)

    if w>1e11:
        max_step = 5.0e-13
    elif w>1e9:
        max_step = 1.0e-11
    else:
        max_step = 1.0e-11

    rate0.I_func = I_func
    rate0.t = time
    rate0.max_step = max_step
    rate0.N0=[1e17*1e6]

    tt, Nt = rate0.run(option='nanoLED')

    t_start = 0.6*time[1]

    deltaP_ant0[i] = rate0.calc_peak_to_peak_power(np.squeeze(tt),
                                                np.squeeze(Nt),
                                                t_start, 'nanoLED', 'Power')

    I_func = lambda t: I_ant1 + I1_ant1*np.sin(w*t)

    rate1.I_func = I_func
    rate1.t = time
    rate1.max_step = max_step
    rate1.N0=[1e17*1e6]

    tt, Nt = rate1.run(option='nanoLED')

    t_start = 0.6*time[1]

    deltaP_ant1[i] = rate1.calc_peak_to_peak_power(np.squeeze(tt),
                                                np.squeeze(Nt),
                                                t_start, 'nanoLED', 'Power')

    I_func = lambda t: I_ant2 + I1_ant2*np.sin(w*t)

    rate2.I_func = I_func
    rate2.t = time
    rate2.max_step = max_step
    rate2.N0 = [1e17*1e6]

    tt, Nt = rate2.run(option='nanoLED')

    t_start = 0.6*time[1]

    deltaP_ant2[i] = rate2.calc_peak_to_peak_power(np.squeeze(tt),
                                                np.squeeze(Nt),
                                                t_start, 'nanoLED', 'Power')

    I_func = lambda t: I_ant3 + I1_ant3*np.sin(w*t)

    rate3.I_func = I_func
    rate3.t = time
    rate3.max_step = max_step
    rate3.N0 = [1e17*1e6]

    tt, Nt = rate3.run(option='nanoLED')

    t_start = 0.6*time[1]

    deltaP_ant3[i] = rate3.calc_peak_to_peak_power(np.squeeze(tt),
                                                np.squeeze(Nt),
                                                t_start, 'nanoLED', 'Power')

deltaP_ant0 = COMM.gather(deltaP_ant0, root=0)
deltaP_ant1 = COMM.gather(deltaP_ant1, root=0)
deltaP_ant2 = COMM.gather(deltaP_ant2, root=0)
deltaP_ant3 = COMM.gather(deltaP_ant3, root=0)
simulations_curr = COMM.gather(simulations_curr, root=0)

if RANK == 0:
    deltaP_ant0_new = np.zeros((ns_master.size, w_master.size))
    deltaP_ant1_new = np.zeros((ns_master.size, w_master.size))
    deltaP_ant2_new = np.zeros((ns_master.size, w_master.size))
    deltaP_ant3_new = np.zeros((ns_master.size, w_master.size))
    for i in range(len(simulations_curr)):
        for j in range(len(simulations_curr[i])):
            w = simulations_curr[i][j][1]
            index = simulations_curr[i][j][0]

            I_ant0 = I_master[index, 0]
            I_ant1 = I_master[index, 1]
            I_ant2 = I_master[index, 2]
            I_ant3 = I_master[index, 3]

            w_index = np.where(w_master==w)
            I_index = np.where(I_master[:,0]==I_ant0)

            deltaP_ant0_new[I_index, w_index] = deltaP_ant0[i][j]
            deltaP_ant1_new[I_index, w_index] = deltaP_ant1[i][j]
            deltaP_ant2_new[I_index, w_index] = deltaP_ant2[i][j]
            deltaP_ant3_new[I_index, w_index] = deltaP_ant3[i][j]


    opttransfer = Eg/q

    outfile = 'data_WDM_transfer3'
    np.savez(outfile, omega=omega, V=V,
             deltaP_ant0=deltaP_ant0_new,
             deltaP_ant1=deltaP_ant1_new, deltaP_ant2=deltaP_ant2_new,
             deltaP_ant3=deltaP_ant3_new,
             I_master = I_master, w_master=w_master,
             ns_master=ns_master,opttransfer=opttransfer)

    fraction = []
    for nnnn in ns_master:
        wdm.find_collected_power(nnnn)
        fraction.append(wdm.fractions)

    outfile = 'data_WDM_transfer3_fracs'
    np.savez(outfile, fractions=fraction)
