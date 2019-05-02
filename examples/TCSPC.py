from math import pi
import numpy as np
from physical_constants import *
import active_material
import LED
import rate_equations
import parallel_toolkit

SIZE = parallel_toolkit.SIZE
RANK = parallel_toolkit.RANK
COMM = parallel_toolkit.COMM


n = 3.5
Egw = 0.755 * q
Ep = 25.7 * q
Nd = 0.0
Na = 2.0e18 * 1e6
Lz = 4.75e-9
M = (m0/6)*Ep
mw = 0.041*m0
mw_lh = 0.0503*m0
mw_hh = 0.46*m0

mb = 0.065*m0
mb_lh = 0.087*m0
mb_hh = 0.46*m0

Egb = 1.03*q

delEc = 0.4

A = 1.0/(1e-6)
C = 1e-40

vs = 0.5e3 * 1.0e-2 # assuming low temp
#vs = 0.0
surface = 2.0/(100e-9) # surface to volume ratio

omega = np.linspace(0.4, 3.0, num=3000)*q/hbar
DF_max = 1.2*q
DF_dis = 1.2*q/600
T = 300.0

QW = active_material.QuantumWell(
    omega,
    DF_max,
    DF_dis,
    T,
    n,
    mw,
    mw_lh,
    mw_hh,
    Ep,
    M,
    Egw,
    Na,
    Nd,
    Lz,
    Egb,
    mb,
    mb_lh,
    mb_hh,
    delEc,
    A,
    C
)


QW.build()
f_supp = 0.1
#Fx = f_supp*np.ones(omega.size)
#Fy = f_supp*np.ones(omega.size)
#Fz = f_supp*np.ones(omega.size)
Fx = f_supp*np.ones(omega.size)
Fy = f_supp*np.ones(omega.size)
Fz = f_supp*np.ones(omega.size)
antenna = LED.Antenna(omega, Fx, Fy, Fz)
#antenna = LED.Antenna(omega, Fx)

nanoLED = LED.nanoLED_QW(QW, antenna, vs, surface)
nanoLED.build()

print 'build_complete'


omegas = QW.omega
N = QW.N
rspon = nanoLED.rspon_ant_broadened
Rspon = nanoLED.Rspon_ant
Rnr = nanoLED.Rnr
Rnr_ant = nanoLED.Rnr_ant
if RANK==0: print N/Rnr
if RANK==0: print N/Rnr_ant
if RANK==0: print N[-1]

I_func = lambda t: np.sin(2*pi*t/(1e-8))
reqns = rate_equations.RateEquations(nanoLED, [0, 1.0e-6], I_func, V=1e-18,
                                     N0=[1.0e23], etai=1, max_step = 1e-12)


LED_option = 'nanoLED'
#ns_master = 1.0e6*np.logspace(16, 19, num=21)
ns_master = 1.0e6 * np.logspace(16, 19, num=21)

partition = parallel_toolkit.parallel_partition(SIZE)

Is = np.zeros(len(partition))

for i in range(len(partition)):
    par = partition[i]
    nn = ns_master[par]
    print nn
    II, __ = reqns.find_desired_Idc([0, 5.0e-7], 5.0e-12, nn, [1.0e-8, 5.0e-1])

    Is[i] = II


Is = COMM.gather(Is, root=0)

I_master = parallel_toolkit.create_master_array(ns_master.size, Is)

I_master = COMM.bcast(I_master, root=0)

ts = []
Nts = []
Is = []

for i in range(len(partition)):
    print('Node {0} is performing simulation {1} of {2}'.format(RANK, i+1,
                                                len(partition)))
    par = partition[i]
    II = I_master[par]

    curr_func = lambda tt: 0.0 if tt>=2.0e-7 else II
    
    reqns.t = [0, 5.0e-7]
    reqns.I_func = curr_func
    reqns.max_step = 1.0e-12

    tt, Nt = reqns.run(option = 'nanoLED')
    
    #Is.append(curr_func(tt))
    ts.append(tt)
    Nts.append(Nt)

#Is = COMM.gather(Is, root=0)
ts = COMM.gather(ts, root=0)
Nts = COMM.gather(Nts, root=0)

partition = COMM.gather(partition, root=0)

if RANK == 0:
    outfile = 'data_TCSPC_InGaAs_QW_test6_2e18_10xSupp_normalSRV_normalC.npz'
    np.savez(outfile, ts=ts, Nts=Nts, partition=partition, omega=omegas, N=N,
             rspon=rspon, Rspon=Rspon)
