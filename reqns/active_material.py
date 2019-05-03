#!/usr/bin/env python
"""
Provides class ActiveMaterial and subclasses Bulk and QuantumWell.

Bulk and QuantumWell classes can be used to calculate carrier
concentrations and material gain spectra. These are indexed by quasi-Fermi level
differences (applied voltage).
"""

from math import pi, ceil
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy.optimize import fsolve

from physical_constants import h, hbar, c, q, eps0, m0, k
import misc

__author__ = "Sean Hooten"
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class ActiveMaterial(object):
    """Superclass for active radiative materials.

    Includes properties and methods that are common to calculating gain in
    active materials.
    """

    __metaclass__ = ABCMeta

    def __init__(self, Na, Nd, T):
        self._Na = Na
        self._Nd = Nd
        self._T = T
        self._correct_build = False

    @property
    def Na(self):
        return self._Na

    @Na.setter
    def Na(self, val):
        self._Na = val
        self._correct_build = False

    @property
    def Nd(self):
        return self._Nd

    @Nd.setter
    def Nd(self, val):
        self._Nd = val
        self._correct_build = False

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        self._T = val
        self._correct_build = False

    @abstractproperty
    def ni(self):
        pass

    @property
    def n0(self):
        return self._n0

    @property
    def p0(self):
        return self._p0

    def fermi_dirac_function(self, E, F):
        kT = k*self.T
        f = 1 / (1 + np.exp( (E - F) / kT ))
        return f

    def fermi_inversion_function(self, E1, Fv, E2, Fc):
        f2 = self.fermi_dirac_function(E2, Fc)
        f1 = self.fermi_dirac_function(E1, Fv)
        return f2 - f1

#    def fermi_emission_factor(self, E1, Fv, E2, Fc, T):
#        f1 = 1 - self.fermi_dirac_function(E1, Fv, T)
#        f2 = self.fermi_dirac_function(E2, Fc, T)
#        return f2 * f1

    def calc_n0_p0(self):
        # calculate n and p at zero bias
        # requires that self.ni is defined

        if self.Nd - self.Na > 10*self.ni:
            n0 = self.Nd-self.Na
            p0 = self.ni**2/n0
        elif self.Na - self.Nd > 10*self.ni:
            p0 = self.Na-self.Nd
            n0 = self.ni**2/p0
        else:
            n0 = 0.5*(self.Nd - self.Na) + \
                    np.sqrt((0.5*(self.Nd-self.Na))**2+self.ni**2)

            p0 = 0.5*(self.Na - self.Nd) + \
                    np.sqrt((0.5*(self.Na-self.Nd))**2+self.ni**2)

        self._n0 = n0
        self._p0 = p0

class Bulk(ActiveMaterial):
    """TO DO:
        include support for light and heavy hole bands
    """

    def __init__(self, omega, DF_max, DF_dis, Na, Nd, T, n, me, mh, M, Eg, A, C):
        super(Bulk, self).__init__(Na, Nd, T)
        # Required Material Data
        self._n = n
        self._M = M
        self._Eg = Eg

        self._me = me
        self._mh = mh
        self._mr = me * mh / (me + mh)

        self._Nc = 2 * (2 * pi * me * k *self.T / h**2)**(1.5)
        self._Nv = 2 * (2 * pi * mh * k * self.T / h**2)**(1.5)

        self._A = A
        self._C = C

        # Required User Inputs
        self._DF_max = DF_max # assume minimum 0, max defined by source
        self._omega = omega # must choose sufficiently discretized and large
        self._DF_dis = DF_dis # (roughly) discretization of DF points

        # To be calculated with self.build()
        self._DF = None
        self._ni = None
        self._n0 = None
        self._p0 = None
        self._Fc = None
        self._Fv = None
        self._N = None
        self._P = None
        self._E1 = None
        self._E2 = None
        self._rho = None
        self._fg = None
        self._gain = None
        self._gain_broadened = None

        self._correct_build = False

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val
        self._correct_build = False

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, val):
        self._M = val
        self._correct_build = False

    @property
    def Eg(self):
        return self._Eg

    @Eg.setter
    def Eg(self, val):
        self._Eg = val
        self._correct_build = False

    @property
    def me(self):
        return self._me

    @me.setter
    def me(self, val):
        self._me = val
        self._correct_build = False

    @property
    def mh(self):
        return self._mh

    @mh.setter
    def mh(self, val):
        self._mh = val
        self._correct_build = False

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, val):
        self._A = val
        self._correct_build = False

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        self._C = val
        self._correct_build = False

    @property
    def mr(self):
        return self._mr

    @property
    def Nc(self):
        return self._Nc

    @property
    def Nv(self):
        return self._Nv

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        self._omega = val
        self._correct_build = False

    @property
    def DF_max(self):
        return self._DF_max

    @DF_max.setter
    def DF_max(self, val):
        self._DF_max = val
        self._correct_build = False

    @property
    def DF_dis(self):
        return self._DF_dis

    @DF_dis.setter
    def DF_dis(self, val):
        self._DF_dis = val
        self._correct_build = False

    @property
    def DF(self):
        return self._DF

    @property
    def ni(self):
        return self._ni

    @property
    def Fc(self):
        return self._Fc

    @property
    def Fv(self):
        return self._Fv

    @property
    def N(self):
        return self._N

    @property
    def P(self):
        return self._P

    @property
    def rho(self):
        return self._rho

    @property
    def fg(self):
        return self._fg

    @property
    def gain(self):
        return self._gain

    @property
    def gain_broadened(self):
        return self._gain_broadened

    @property
    def E1(self):
        return self._E1

    @property
    def E2(self):
        return self._E2

    @property
    def correct_build(self):
        return self._correct_build

    def build(self, broaden_gain = False, tau_m = 0.1e-12):
        self._mr = self.me * self.mh / (self.me + self.mh)
        self._Nc = 2 * (2 * pi * self.me * k * self.T / h**2)**(1.5)
        self._Nv = 2 * (2 * pi * self.mh * k * self.T / h**2)**(1.5)

        self._calc_N_P()
        self._calc_DOSjoint()
        self._calc_E1()
        self._calc_E2()

        fg = np.zeros((self.omega.size, self.DF.size))
        gain = np.zeros((self.omega.size, self.DF.size))

        for i in range(self.DF.size):
            fg[:, i] = self._fermi_inversion_function(self.Fv[i], self.Fc[i])
            gain[:, i] = self._calc_gain(fg[:, i])

        self._fg = fg

        # each polarization sees the same gain
        self._gain = np.zeros((self.omega.size, self.DF.size, 3))
        self._gain[:,:,0] = gain
        self._gain[:,:,1] = gain
        self._gain[:,:,2] = gain

        if broaden_gain:
            gain_broadened = np.zeros((self.omega.size, self.DF.size))
            for i in range(self.DF.size):
                gain_broadened[:,i] = \
                    misc.lineshape_broadening(self._omega, self._gain[:,i,0], tau_m)
            self._gain_broadened = np.zeros((self.omega.size, self.DF.size, 3))
            self._gain_broadened[:,:,0] = gain_broadened
            self._gain_broadened[:,:,1] = gain_broadened
            self._gain_broadened[:,:,2] = gain_broadened

        self._correct_build = True

    def _fermi_inversion_function(self, Fv, Fc):
        E1 = self.E1
        E2 = self.E2
        fg = super(Bulk, self).fermi_inversion_function(E1, Fv, E2, Fc)
        return fg

    def _calc_gain(self, fg):
        rho = self.rho
        C0 = pi*q**2 / (self.n * c * eps0 * m0**2 * self.omega)
        gain = C0 * rho * self.M * fg
        return gain

    def _calc_N_P(self):
        num = int(ceil((self._Eg + self._DF_max) / self._DF_dis))
        Fcs = np.linspace(-self.DF_max, self.Eg + self.DF_max, num=num)
        Fvs = np.linspace(-self.DF_max, self.Eg + self.DF_max, num=num)

        beta = 1/(k*self.T)
        N = np.zeros(num)
        P = np.zeros(num)

        N = self.Nc * misc.fermi_dirac_integral(k=0.5, phi=beta*(Fcs-self.Eg))
        P = self.Nv * misc.fermi_dirac_integral(k=0.5, phi=beta*(-Fvs))

        #N = self.Nc * np.exp(beta*(Fcs-self.Eg))
        #P = self.Nv * np.exp(beta*(-Fvs))

        # when Fc = Fv = Fref, n=n0, p=p0
        ref_ni = np.argmin(np.abs(N-P)) # only works when Fcs and Fvs arrays are the same
        self._ni = N[ref_ni] # check this

        self.calc_n0_p0()

        nprime = N-self.n0
        pprime = P-self.p0

        refN = np.argmin(np.abs(nprime))
        refP = np.argmin(np.abs(pprime))

        FrefN = Fcs[refN]
        FrefP = Fvs[refP]
        # might need to check whether these are close enough

        count = np.min([num-refN, refP])
        DF = np.zeros(count)
        NN = np.zeros(count)
        PP = np.zeros(count)
        Fc = np.zeros(count)
        Fv = np.zeros(count)

        if self.p0 >= self.n0:
            for i in range(count):
                Fc[i] = Fcs[refN+i]
                #print nprime[refN+i], N[refN+i]
                #import matplotlib.pyplot as plt
                #f = plt.figure()
                #ax1 = f.add_subplot(211)
                #ax1.semilogy(Fvs/q, np.abs(nprime[refN+i]-pprime))

                #ax2 = f.add_subplot(212)
                #ax2.semilogy(Fvs/q, nprime)
                #plt.show()

                Fv[i] = np.interp(0.0, nprime[refN+i]-pprime, Fvs)
                #print Fv[i]/q
                DF[i] = Fc[i] - Fv[i]
                NN[i] = N[refN+i]
                PP[i] = np.interp(Fv[i], Fvs, P)

        else:
            for i in range(count):
                Fv[i] = Fvs[refP-i]
                Fc[i] = np.interp(0.0,nprime-pprime[refP-i], Fcs)
                DF[i] = Fc[i] - Fv[i]
                PP[i] = P[refP-i]
                NN[i] = np.interp(Fc[i], Fcs, N)

        index = np.argmin(np.abs(DF-self.DF_max))
        DF = DF[:index]
        NN = NN[:index]
        PP = PP[:index]

        Fc = Fc[:index]
        Fv = Fv[:index]

        self._N = NN
        self._P = PP
        self._DF = DF

        self._Fc = Fc
        self._Fv = Fv

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(DF/q, NN/1e6, '-o')
        #ax.semilogy(DF/q, PP/1e6, '-o')
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'N, P (cm$^{-3}$)')
        #plt.show()

    def _calc_DOSjoint(self):
        args = hbar*self.omega - self.Eg
        arg = np.array([x if x>=0 else 0 for x in args]) # test for negatives
        rho = (1 / (2*pi**2)) * (2*self.mr/(hbar**2))**(1.5) * np.sqrt(arg)

        ######## WHAT IS THIS? ########
        rho = 4*rho # factor of 4 for spin degeneracy? Exciton effects?
        ###############################

        self._rho = rho

    def _calc_E1(self):
        E1 = -1 * (hbar*self.omega - self.Eg) * self.mr / self.mh
        self._E1 = E1

    def _calc_E2(self):
        E2 = self.Eg + (hbar*self.omega - self.Eg) * self.mr / self.me
        self._E2 = E2

class QuantumWell(ActiveMaterial):
    def __init__(self, omega, DF_max, DF_dis, Na, Nd, T, n, M, Egw, Lz, mw,
                 mw_lh, mw_hh, A, C, Egb = False, mb = False,
                 mb_lh = False, mb_hh = False, delEc = False):

        super(QuantumWell, self).__init__(Na, Nd, T)
        # Required Material Data
        # MAKE SURE ALL VALUES ARE SI

        self._n = n
        self._M = M
        self._Egw = Egw # quantum well bandgap
        self._Lz = Lz # quantum well depth

        self._mw = mw
        self._mw_hh = mw_hh
        self._mw_lh = mw_lh

        self._A = A
        self._C = C

        self._mr_hh = mw_hh*mw/(mw_hh + mw)
        self._mr_lh = mw_lh*mw/(mw_lh + mw)

        self._Nc = mw * k * T / (pi * hbar**2 * Lz)
        self._Nv_hh = mw_hh * k * T / (pi * hbar**2 * Lz)
        self._Nv_lh = mw_lh * k * T / (pi * hbar**2 * Lz)

        # Barrier Material Data (Optional)
        self._finite_well = False
        self._Egb = Egb
        self._mb = mb
        self._mb_lh = mb_lh
        self._mb_hh = mb_hh
        self._delEc = delEc

        #self.interface_recombination_velocity = None

        # Required User Inputs
        self._DF_max = DF_max # assume minimum 0, max defined by source
        self._omega = omega #must choose sufficiently discretized and large
        self._DF_dis = DF_dis # (roughly) discretization of DF points

        # To be calculated with self.build()
        self._Ew_e = None
        self._Ew_lh = None
        self._Ew_hh = None

        self._DF = None
        self._ni = None
        self._n0 = None
        self._p0 = None
        self._Fc = None
        self._Fv = None
        self._N = None
        self._P = None
        self._rho_lh = None
        self._rho_hh = None

        self._E1_lh = None
        self._E2_lh = None
        self._E1_hh = None
        self._E2_hh = None

        self._fg_hh = None
        self._fg_lh = None
        self._gain = None
        self._gain_lh = None
        self._gain_hh = None
        self._gain_broadened = None

        self._correct_build = False

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val
        self._correct_build = False

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, val):
        self._M = val
        self._correct_build = False

    @property
    def Egw(self):
        return self._Egw

    @Egw.setter
    def Egw(self, val):
        self._Egw = val
        self._correct_build = False

    @property
    def Lz(self):
        return self._Lz

    @Lz.setter
    def Lz(self, val):
        self._Lz = val
        self._correct_build = False

    @property
    def mw(self):
        return self._mw

    @mw.setter
    def mw(self, val):
        self._mw = val
        self._correct_build = False

    @property
    def mw_hh(self):
        return self._mw_hh

    @mw_hh.setter
    def mw_hh(self, val):
        self._mw_hh = val
        self._correct_build = False

    @property
    def mw_lh(self):
        return self._mw_lh

    @mw_lh.setter
    def mw_lh(self, val):
        self._mw_lh = val
        self._correct_build = False

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, val):
        self._A = val
        self._correct_build = False

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, val):
        self._C = val
        self._correct_build = False

    @property
    def Egb(self):
        return self._Egb

    @Egb.setter
    def Egb(self, val):
        self._Egb = val
        self._correct_build = False

    @property
    def mb(self):
        return self._mb

    @mb.setter
    def mb(self, val):
        self._mb = val
        self._correct_build = False

    @property
    def mb_lh(self):
        return self._mb_lh

    @mb_lh.setter
    def mb_lh(self, val):
        self._mb_lh = val
        self._correct_build = False

    @property
    def mb_hh(self):
        return self._mb_hh

    @mb_hh.setter
    def mb_hh(self, val):
        self._mb_hh = val
        self._correct_build = False

    @property
    def delEc(self):
        return self._delEc

    @delEc.setter
    def delEc(self, val):
        self._delEc = val
        self._correct_build = False

    @property
    def mr_lh(self):
        return self._mr_lh

    @property
    def mr_hh(self):
        return self._mr_hh

    @property
    def Nc(self):
        return self._Nc

    @property
    def Nv_hh(self):
        return self._Nv_hh

    @property
    def Nv_lh(self):
        return self._Nv_lh

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        self._omega = val
        self._correct_build = False

    @property
    def DF_max(self):
        return self._DF_max

    @DF_max.setter
    def DF_max(self, val):
        self._DF_max = val
        self._correct_build = False

    @property
    def DF_dis(self):
        return self._DF_dis

    @DF_dis.setter
    def DF_dis(self, val):
        self._DF_dis = val
        self._correct_build = False

    @property
    def Ew_e(self):
        return self._Ew_e

    @property
    def Ew_lh(self):
        return self._Ew_lh

    @property
    def Ew_hh(self):
        return self._Ew_hh

    @property
    def DF(self):
        return self._DF

    @property
    def ni(self):
        return self._ni

    @property
    def Fc(self):
        return self._Fc

    @property
    def Fv(self):
        return self._Fv

    @property
    def N(self):
        return self._N

    @property
    def P(self):
        return self._P

    @property
    def rho_lh(self):
        return self._rho_lh

    @property
    def rho_hh(self):
        return self._rho_hh

    @property
    def E1_lh(self):
        return self._E1_lh

    @property
    def E2_lh(self):
        return self._E2_lh

    @property
    def E1_hh(self):
        return self._E1_hh

    @property
    def E2_hh(self):
        return self._E2_hh

    @property
    def fg_lh(self):
        return self._fg_lh

    @property
    def fg_hh(self):
        return self._fg_hh

    @property
    def gain_lh(self):
        return self._gain_lh

    @property
    def gain_hh(self):
        return self._gain_hh

    @property
    def gain(self):
        return self._gain

    @property
    def gain_broadened(self):
        return self._gain_broadened

    @property
    def correct_build(self):
        return self._correct_build

    def build(self, broaden_gain = False, tau_m = 0.1e-12):
        self._mr_hh = 1 / (1/self._mw + 1/self._mw_hh)
        self._mr_lh = 1 / (1/self._mw + 1/self._mw_lh)

        self._Nc = self.mw * k * self.T / (pi * hbar**2 * self.Lz)
        self._Nv_lh = self.mw_lh * k * self.T / (pi * hbar**2 * self.Lz)
        self._Nv_hh = self.mw_hh * k * self.T / (pi * hbar**2 * self.Lz)

        self.check_barrier_params()
        self.calc_eigen_energies()
        self._calc_N_P()
        self._calc_DOSjoint()
        self._calc_E1()
        self._calc_E2()

        fg_lh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))
        fg_hh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))

        for i in range(self.DF.size):
            for j in range(self.Ew_e.size):
                for n in range(self.omega.size):
                    fg_lh[n, i, j] = self.fermi_inversion_function(self.E1_lh[n,j], self.Fv[i],
                                                               self.E2_lh[n,j], self.Fc[i])
                    fg_hh[n, i, j] = self.fermi_inversion_function(self.E1_hh[n,j], self.Fv[i],
                                                               self.E2_hh[n,j], self.Fc[i])

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #f2 = plt.figure()
        #ax = f.add_subplot(111)
        #ax2 = f2.add_subplot(111)
        #ax.plot(self.omega*hbar/q, fg_lh[:,:,0])
        #ax2.plot(self.omega*hbar/q, fg_hh[:,:,0])
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'$F_c, F_v$')
        #plt.show()

        self._fg_lh = fg_lh
        self._fg_hh = fg_hh

        self._calc_gain2()

        if broaden_gain:
            gain_broadened = np.zeros((self.omega.size, self.DF.size, 3))
            for i in range(self.DF.size):
                for j in range(3):
                    gain_broadened[:,i,j] = \
                        misc.lineshape_broadening(self._omega, self._gain[:,i,j], tau_m)
            self._gain_broadened = gain_broadened

        self._correct_build = True

    def check_barrier_params(self):
        barrier_params = [self.Egb, self.mb, self.mb_lh, self.mb_hh,
                          self.delEc]
        if all(barrier_params):
            self._finite_well = True
            print "Barrier parameters set, will solve for finite QW barriers"
        elif any(barrier_params):
            print "Not all barrier parameters are set, will assume infinite QW barriers"
            self._finite_well = False
        else:
            print "Barrier parameters not set, will assume infinite QW barriers"
            self._finite_well = False

    def calc_eigen_energies(self):
        #Vo_e = (self.Egb - self.Egw) * delEc / q # this is a voltage
        #Vo_h = (self.Egb - self.Egw) / q  - Vo_e

        if self._finite_well:
            self.calc_eigen_energies_finite()
        else:
            max_energy = hbar*self.omega[-1]

            Ew_e = []
            Ew_lh = []
            Ew_hh = []

            difference = 0
            nn = 1

            # need to find all quantum well energy levels
            # to-do: truncate at conduction band offset
            while max_energy - difference >= 0:
            #Currently infinite barrier energies considered
                Ee =  nn**2 * hbar**2 * pi**2 / (2 * self.mw * self.Lz**2)
                Ehh = nn**2 * hbar**2 * pi**2 / (2 * self.mw_hh * self.Lz**2)
                Elh = nn**2 * hbar**2 * pi**2 / (2 * self.mw_lh * self.Lz**2)

                Ew_e.append(Ee)
                Ew_lh.append(Elh)
                Ew_hh.append(Ehh)

                difference_lh = Ee + self.Egw + Elh
                difference_hh = Ee + self.Egw + Ehh
                difference = np.min([difference_lh, difference_hh])

                nn += 1

            # note that all quantized energies are positive, need to adjust based on
            # coordinate system

            self._Ew_e = np.array(Ew_e)
            self._Ew_lh = np.array(Ew_lh)
            self._Ew_hh = np.array(Ew_hh)

            print self.Ew_e / q
            print self.Ew_lh / q
            print self.Ew_hh / q

    def calc_eigen_energies_finite(self):
        max_energy = hbar*self.omega[-1]

        Ew_e = []
        Ew_lh = []
        Ew_hh = []

        difference = 0

        # delEc = V0_e / (V0_e + V0_b)
        V0_e = (self.Egb - self.Egw)*self.delEc
        V0_h = (self.Egb - self.Egw) - V0_e

        u_lh = np.sqrt(self.mb_lh * self.Lz**2 * V0_h / (2 * hbar**2))
        u_hh = np.sqrt(self.mb_hh * self.Lz**2 * V0_h / (2 * hbar**2))
        u_e = np.sqrt(self.mb * self.Lz**2 * V0_e / (2 * hbar**2))

        num_lh = ceil(2*u_lh/pi * np.sqrt(self.mb_lh/self.mw_lh))
        num_hh = ceil(2*u_hh/pi * np.sqrt(self.mb_hh/self.mw_hh))
        num_e = ceil(2*u_e/pi * np.sqrt(self.mb/self.mw))

        def even_solution(x, V0, mb, mw):
            u2 = mw * self.Lz**2 * V0 / (2 * hbar**2)
            y1 = np.sqrt(u2 - x**2)
            y2 = np.sqrt(mw / mb) * x * np.tan(x)
            return y1 - y2

        def odd_solution(x, V0, mb, mw):
            u2 = mw * self.Lz**2 * V0 / (2 * hbar**2)
            y1 = np.sqrt(u2 - x**2)
            y2 = -1 * np.sqrt(mw / mb) * x / np.tan(x)
            return y1 - y2

        num_v_subbands = np.max([num_lh, num_hh])
        num = int(np.min([num_e, num_v_subbands]))

        for i in range(num):
            u = np.sqrt(self.mb * self.Lz**2 * V0_e / (2 * hbar**2))
            if i % 2 == 1:
                guess = np.sqrt(self.mw/self.mb) * ((i+1) * pi/2 - pi/4)
                kL_over_2 = fsolve(even_solution, guess, args=(V0_e, self.mb, self.mw))
            else:
                guess = np.sqrt(self.mw/self.mb) * ((i+1) * pi/2 - pi/4)
                kL_over_2 = fsolve(odd_solution, guess, args=(V0_e, self.mb, self.mw))
            Ee = 2 * hbar**2 * kL_over_2**2 / (self.mw * self.Lz**2)
            Ew_e.append(Ee)

            u = np.sqrt(self.mb_hh * self.Lz**2 * V0_h / (2 * hbar**2))
            if i % 2 == 1:
                guess = np.sqrt(self.mw_hh/self.mb_hh) * ((i+1) * pi/2 - pi/4)
                kL_over_2 = fsolve(even_solution, guess, args=(V0_h,
                                                               self.mb_hh,
                                                               self.mw_hh))
            else:
                guess = np.sqrt(self.mw_hh/self.mb_hh) * ((i+1) * pi/2 - pi/4)
                kL_over_2 = fsolve(odd_solution, guess, args=(V0_h,
                                                              self.mb_hh,
                                                              self.mw_hh))
            Ehh = 2 * hbar**2 * kL_over_2**2 / (self.mw_hh * self.Lz**2)
            Ew_hh.append(Ehh)

            if num_lh >= i: # it's possible that there are more conduction and
                            # heavy hole subbands, need to correct for this
                u = np.sqrt(self.mb_lh * self.Lz**2 * V0_h / (2 * hbar**2))
                if i % 2 == 1:
                    guess = np.sqrt(self.mw_lh/self.mb_lh) * ((i+1) * pi/2 - pi/4)
                    kL_over_2 = fsolve(even_solution, guess, args=(V0_h,
                                                                   self.mb_lh,
                                                                   self.mw_lh))
                else:
                    guess = np.sqrt(self.mw_lh/self.mb_lh) * ((i+1) * pi/2 - pi/4)
                    kL_over_2 = fsolve(odd_solution, guess, args=(V0_h,
                                                                  self.mb_lh,
                                                                  self.mw_lh))
                Elh = 2 * hbar**2 * kL_over_2**2 / (self.mw_lh * self.Lz**2)
                Ew_lh.append(Elh)
            else:
                Elh = 1000.0
                Ew_lh.append(Elh) # Dummy for calculations

        self._Ew_e = np.array(Ew_e)
        self._Ew_lh = np.array(Ew_lh)
        self._Ew_hh = np.array(Ew_hh)

        print self.Ew_e / q
        print self.Ew_lh / q
        print self.Ew_hh / q

    def _calc_N_P(self):
        num = int(ceil((self.Egw+self.Ew_e[-1]+self.Ew_lh[-1]+2*self.DF_max) / self.DF_dis))
        Fcs = np.linspace(-self.Ew_lh[-1]-self.DF_max, self.Egw+self.Ew_e[-1]+self.DF_max, num=num)
        Fvs = np.linspace(-self.Ew_lh[-1]-self.DF_max, self.Egw+self.Ew_e[-1]+self.DF_max, num=num)

        beta = 1/(k*self.T)
        N = np.zeros(num)
        P = np.zeros(num)

        for i in range(self.Ew_e.size):
            N += self.Nc * misc.fermi_dirac_integral(k=0,phi=beta*(Fcs-self.Ew_e[i]-self.Egw))

            P += self.Nv_lh * misc.fermi_dirac_integral(k=0,phi=beta*(-Fvs-self.Ew_lh[i]))
            P += self.Nv_hh * misc.fermi_dirac_integral(k=0,phi=beta*(-Fvs-self.Ew_hh[i]))

        ########
        #CHECK THIS!
        ########


        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(Fcs/q, N/1e6, '-o')
        #ax.semilogy(Fvs/q, P/1e6, '-o')
        ##plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'N, P (cm$^{-3}$)')
        #plt.show()
        ## might need to check this for density of states of each subband


        # when Fc = Fv = Fref, n=n0, p=p0
        #import matplotlib.pyplot as plt
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(Fcs/q, N)
        #ax.semilogy(Fvs/q, P)
        #ax.semilogy(Fcs/q, np.abs(N-P))
        #plt.show()

        ref_ni = np.argmin(np.abs(N-P)) # only works when Fcs and Fvs arrays are the same
        self._ni = N[ref_ni] # check this

        self.calc_n0_p0()

        nprime = N-self.n0
        pprime = P-self.p0

        refN = np.argmin(np.abs(nprime))
        refP = np.argmin(np.abs(pprime))

        FrefN = Fcs[refN]
        FrefP = Fvs[refP]
        # might need to check whether these are close enough

        count = np.min([num-refN, refP])
        DF = np.zeros(count)
        NN = np.zeros(count)
        PP = np.zeros(count)
        Fc = np.zeros(count)
        Fv = np.zeros(count)

        if self.p0 >= self.n0:
            for i in range(count):
                Fc[i] = Fcs[refN+i]
                Fv[i] = np.interp(0.0, nprime[refN+i]-pprime, Fvs)
                DF[i] = Fc[i] - Fv[i]
                NN[i] = N[refN+i]
                PP[i] = np.interp(Fv[i], Fvs, P)

        else:
            for i in range(count):
                Fv[i] = Fvs[refP-i]
                Fc[i] = np.interp(0.0,nprime-pprime[refP-i], Fcs)
                DF[i] = Fc[i] - Fv[i]
                PP[i] = P[refP-i]
                NN[i] = np.interp(Fc[i], Fcs, N)

        index = np.argmin(np.abs(DF-self.DF_max))
        DF = DF[:index]
        NN = NN[:index]
        PP = PP[:index]

        Fc = Fc[:index]
        Fv = Fv[:index]

        self._N = NN
        self._P = PP
        self._DF = DF

        self._Fc = Fc
        self._Fv = Fv

        #if RANK == 0:
        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(DF/q, NN/1e6, '-o')
        #ax.semilogy(DF/q, PP/1e6, '-o')
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'N, P (cm$^{-3}$)')
        #plt.show()

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(Fc/q, NN/1e6, '-o')
        #ax.semilogy(Fv/q, PP/1e6, '-o')
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'N, P (cm$^{-3}$)')
        #plt.show()

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.plot(DF/q, Fc/q, '-o')
        #ax.plot(DF/q, Fv/q, '-o')
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'$F_c, F_v$')
        #plt.show()

    def _calc_DOSjoint(self):
        self._rho_hh = self.mr_hh / (pi * hbar**2 * self.Lz)
        self._rho_lh = self.mr_lh / (pi * hbar**2 * self.Lz)
        # THIS IS NOT A SPECTRUM

    def _calc_E1(self):
        Eg = self.Egw

        Ew_e = self.Ew_e
        Ew_hh = self.Ew_hh
        Ew_lh = self.Ew_lh

        mr_hh = self.mr_hh
        mr_lh = self.mr_lh

        mw_hh = self.mw_hh
        mw_lh = self.mw_lh

        E1_hh = np.zeros((self.omega.size, Ew_e.size))
        E1_lh = np.zeros((self.omega.size, Ew_e.size))

        for i in range(Ew_e.size):
            E1_hh[:,i] = -Ew_hh[i] - mr_hh / mw_hh * (hbar * self.omega - (Ew_e[i] + Eg + Ew_hh[i]))
            E1_lh[:,i] = -Ew_lh[i] - mr_lh / mw_lh * (hbar * self.omega - (Ew_e[i] + Eg + Ew_lh[i]))

        # this is an array
        self._E1_hh = E1_hh
        self._E1_lh = E1_lh

    def _calc_E2(self):
        Eg = self.Egw

        Ew_e = self.Ew_e
        Ew_hh = self.Ew_hh
        Ew_lh = self.Ew_lh

        mr_hh = self.mr_hh
        mr_lh = self.mr_lh

        mw = self.mw

        E2_hh = np.zeros((self.omega.size, Ew_e.size))
        E2_lh = np.zeros((self.omega.size, Ew_e.size))

        for i in range(Ew_e.size):

            E2_hh[:,i] = Eg + Ew_e[i] + mr_hh / mw * (hbar * self.omega - (Ew_e[i] + Eg + Ew_hh[i]))
            E2_lh[:,i] = Eg + Ew_e[i] + mr_lh / mw * (hbar * self.omega - (Ew_e[i] + Eg + Ew_lh[i]))

        # this is an array
        self._E2_hh = E2_hh
        self._E2_lh = E2_lh

    def _calc_gain2(self):
        rho_lh = self.rho_lh
        rho_hh = self.rho_hh

        fg_lh = self.fg_lh
        fg_hh = self.fg_hh

        C0 = pi * q**2 / (self.n * c * eps0 * m0**2 * self.omega)

        gain_lh = np.zeros((self.omega.size, self.DF.size))
        gain_hh = np.zeros((self.omega.size, self.DF.size))

        gain_lh_pol = np.zeros((self.omega.size, self.DF.size, 3))
        gain_hh_pol = np.zeros((self.omega.size, self.DF.size, 3))

        # note that here we assume the subband selection rule holds, i.e.
        # |I_en^hm| = delta_func(n,m)
        # This can cause inaccuracy issues because it's a simplification of valence band
        # mixing

        for n in range(self.DF.size):
            for i in range(self.omega.size):
                if self.Ew_e.size>1:
                    for j in range(self.Ew_e.size):
                        if hbar*self.omega[i] >= self.Ew_e[j] + self.Ew_lh[j] + self.Egw:
                            cos2 = (self.Ew_e[j]+self.Ew_lh[j])/(hbar*self.omega[i]-self.Egw)
                            coeff = C0[i] * rho_lh * fg_lh[i,n,j] * self.M
                            gain_lh_pol[i,n,0] += coeff * (5.0/4.0-3.0/4.0*cos2)
                            gain_lh_pol[i,n,1] += coeff * (5.0/4.0-3.0/4.0*cos2)
                            gain_lh_pol[i,n,2] += coeff * (1.0/2.0+3.0/2.0*cos2)

                        if hbar*self.omega[i] >= self.Ew_e[j] + self.Ew_hh[j] + self.Egw:
                            cos2 = (self.Ew_e[j]+self.Ew_hh[j])/(hbar*self.omega[i]-self.Egw)
                            coeff = C0[i] * rho_hh * fg_hh[i,n,j] * self.M
                            gain_hh_pol[i,n,0] += coeff * 3.0/4.0 * (1+cos2)
                            gain_hh_pol[i,n,1] += coeff * 3.0/4.0 * (1+cos2)
                            gain_hh_pol[i,n,2] += coeff * 3.0/2.0 * (1-cos2)
                else:
                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_lh[0] + self.Egw:
                        cos2 = (self.Ew_e[0]+self.Ew_lh[0])/(hbar*self.omega[i]-self.Egw)
                        coeff = C0[i] * rho_lh * fg_lh[i,n,0] * self.M
                        gain_lh_pol[i,n,0] += coeff * (5.0/4.0-3.0/4.0*cos2)
                        gain_lh_pol[i,n,1] += coeff * (5.0/4.0-3.0/4.0*cos2)
                        gain_lh_pol[i,n,2] += coeff * (1.0/2.0+3.0/2.0*cos2)

                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_hh[0] + self.Egw:
                        cos2 = (self.Ew_e[0]+self.Ew_hh[0])/(hbar*self.omega[i]-self.Egw)
                        coeff = C0[i] * rho_hh * fg_hh[i,n,0] * self.M
                        gain_hh_pol[i,n,0] += coeff * 3.0/4.0 * (1+cos2)
                        gain_hh_pol[i,n,1] += coeff * 3.0/4.0 * (1+cos2)
                        gain_hh_pol[i,n,2] += coeff * 3.0/2.0 * (1-cos2)

        self._gain_lh = gain_lh_pol
        self._gain_hh = gain_hh_pol
        #self.gain = np.sum(gain_lh_pol, axis=2) + np.sum(gain_hh_pol, axis=2)
        self._gain = gain_lh_pol + gain_hh_pol

def test_QW():
    import matplotlib.pyplot as plt
    n = 3.5
    Egw = 0.755*q
    Ep = 25.7 * q
    Nd = 0.0
    Na = 1.0e17 * 1e6
    Lz = 10.0e-9
    M = (m0/6)*Ep
    mw = 0.041*m0
    mw_lh = 0.0503*m0
    mw_hh = 0.46*m0

    #mb = 0.065*m0
    #mb_lh = 0.087*m0
    #mb_hh = 0.46*m0

    #Egb = 1.03*q

    #delEc = 0.4

    omega = np.linspace(0.5, 2.5, num=2000)*q/hbar
    DF_max = 1.5*q
    DF_dis = 1.5*q/200
    T = 300.0

    #QW = QuantumWell(omega, DF_max, DF_dis, Na, Nd, T, n, M, Egw, Lz, mw, mw_lh, mw_hh,
    #                 Egb=Egb, mb=mb, mb_lh=mb_lh, mb_hh=mb_hh, delEc=delEc)

    QW = QuantumWell(omega, DF_max, DF_dis, Na, Nd, T, n, M, Egw, Lz, mw, mw_lh, mw_hh)

    QW.build(broaden_gain = False)
    print QW.gain_lh.shape
    print QW.gain_hh.shape

    f = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    f4 = plt.figure()
    f5 = plt.figure()
    f6 = plt.figure()
    f7 = plt.figure()
    f8 = plt.figure()
    f9 = plt.figure()
    ax = f.add_subplot(111)
    ax2 = f2.add_subplot(111)
    ax3 = f3.add_subplot(111)
    ax4 = f4.add_subplot(111)
    ax5 = f5.add_subplot(111)
    ax6 = f6.add_subplot(111)
    ax7 = f7.add_subplot(111)
    ax8 = f8.add_subplot(111)
    ax9 = f9.add_subplot(111)

    ax.semilogy(QW.DF/q, QW.N / 1e6)
    ax.semilogy(QW.DF/q, QW.P / 1e6)
    for i in range(0, QW.DF.size, 10):
        #ax.plot(hbar*omega/q, QW.gain_broadened[:,i,0])
        #ax2.plot(hbar*omega/q, QW.gain_broadened[:,i,1])
        #ax3.plot(hbar*omega/q, QW.gain_broadened[:,i,2])
        #ax2.plot(hbar*omega/q, QW.gain[:,i])
        ax4.plot(hbar*omega/q, QW.gain_lh[:,i,0])
        ax5.plot(hbar*omega/q, QW.gain_hh[:,i,0])
        ax6.plot(hbar*omega/q, QW.gain_lh[:,i,1])
        ax7.plot(hbar*omega/q, QW.gain_hh[:,i,1])
        ax8.plot(hbar*omega/q, QW.gain_lh[:,i,2])
        ax9.plot(hbar*omega/q, QW.gain_hh[:,i,2])
    plt.show()

def test_bulk():
    import matplotlib.pyplot as plt

    # GaAs Material Data
    n = 3.62
    Eg = 1.424 * q
    me = 0.063 * m0
    mh = 0.51 * m0
    Ep =  25.7 * q
    M = (m0/6) * Ep
    Nd = 0.0
    Na = 1.0e16 * 1e6
    T = 300.0

    omega = np.linspace(0.5, 3.0, num=2000) * q / hbar
    DF_max = 2.0 * q
    DF_dis = DF_max / 1000

    active_mat = Bulk(omega, DF_max, DF_dis, Na, Nd, T, n, me, mh, M, Eg)
    active_mat.build()

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

if __name__ == "__main__":
    test_QW()
