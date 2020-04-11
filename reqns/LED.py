#!/usr/bin/env python
"""
Provides classes LED, nanoLED, and Antenna as well as function purcell_enhancement.

LED class can be used to calculate spontaneous emission
spectra from large volume/area LED devices, as well as radiative and
nonradiative recombination rates. Antenna class allows one
to specify optical antenna enhancement. nanoLED classes provides the
spontaneous emission spectra after antenna enhancement as well as nonradiative
recombination after including surface recombination.
"""

from math import pi, ceil
import numpy as np
from fdint import fdk

from physical_constants import h, hbar, c, q, eps0, m0, k
import misc
from active_material import Bulk, QuantumWell

__author__ = "Sean Hooten"
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class LED(object):
    def __init__(self, active_mat):
        self._active_mat = active_mat
        self._omega = active_mat.omega
        self._DF = active_mat.DF
        self._T = active_mat.T

        if isinstance(active_mat, Bulk):
            self._mat = 'Bulk'
            self._lifetime = None
        elif isinstance(active_mat, QuantumWell):
            self._mat = 'QuantumWell'
            self._rspon_lh = None
            self._rspon_hh = None

        self._rspon = None
        self._rspon_broadened = None
        self._Rnr = None
        self._Rspon = None
        self._hvRspon = None

        self._correct_build = False

        # Need to make these settable
        #self.vg = c / self.active_mat.n # group velocity
        #self.tau_p = 700 / (self.active_mat.Eg/hbar)
        #self.beta = 0.01
        #self.Gamma = 1

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, val):
        self._T = val
        self._active_mat.T = val
        self._correct_build = False

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        self._omega = val
        self._active_mat.omega = val
        self._correct_build = False

    @property
    def DF(self):
        return self._DF

    @property
    def rspon(self):
        return self._rspon

    @property
    def rspon_lh(self):
        return self._rspon_lh

    @property
    def rspon_hh(self):
        return self._rspon_hh

    @property
    def rspon_broadened(self):
        return self._rspon_broadened

    @property
    def Rnr(self):
        return self._Rnr

    @property
    def Rspon(self):
        return self._Rspon

    @property
    def hvRspon(self):
        return self._hvRspon

    @property
    def correct_build(self):
        return self._correct_build

    def build(self, broaden_rspon = False, tau_m = 0.1e-12):
        assert self._active_mat.correct_build,"Active material not built."

        Rspon = np.zeros((self.DF.size))
        hvRspon = np.zeros((self.DF.size))

        Rnr = np.zeros((self.DF.size))

        self.calc_rspon()
        rspon = self.rspon

        if broaden_rspon:
            rspon_broadened = np.zeros((self.omega.size, self.DF.size, 3))
            for j in range(self.DF.size):
                for n in range(3):
                    rspon_broadened[:,j,n] = \
                        misc.lineshape_broadening(self.omega,rspon[:,j,n], tau_m)
            self._rspon_broadened = rspon_broadened
            rspon = rspon_broadened

        for i in range(self.DF.size):
            Rspon[i] = self.calc_Rspon(rspon[:, i, :])
            Rnr[i] = self.calc_Rnr(self._active_mat.N[i], self._active_mat.P[i])
            hvRspon[i] = self.calc_hvRspon(rspon[:, i, :])

        self._Rspon = Rspon
        self._Rnr = Rnr
        self._hvRspon = hvRspon

        self._correct_build = True

    def calc_rspon(self):
        if self._mat == 'Bulk':
            self.calc_rspon_bulk()
        elif self._mat == 'QuantumWell':
            self.calc_rspon_QW()

    def calc_rspon_bulk(self):
        DF = self.DF
        n = self._active_mat.n
        beta = (1 / (k * self.T))

        gain = self._active_mat.gain

        #rspon = np.zeros((self.omega.size, self.DF.size))
        rspon_broadened = np.zeros((self.omega.size, self.DF.size, 3))
        rspon = np.zeros((self.omega.size, self.DF.size, 3))

        for i in range(self.omega.size):
            coeff = (n**2 * self.omega[i]**2) / (pi**2 * hbar * c**2)
            for j in range(self.DF.size):
                if not hbar*self.omega[i] - DF[j] == 0:
                    rspon[i, j, :] = coeff * gain[i, j, :] * 1 / (1 - np.exp(beta*(hbar * self.omega[i] - DF[j])))

        for i in range(self.omega.size):
            for j in range(self.DF.size):
                if hbar*self.omega[i] - DF[j] == 0:
                    rspon[i, j, :] = 0.5 * (rspon[i-1,j,:] + rspon[i+1,j,:])

        self._rspon = rspon

    def calc_rspon_QW(self):
        DF = self.DF
        n = self._active_mat.n
        beta = (1 / (k * self.T))

        gain_lh = self._active_mat.gain_lh
        gain_hh = self._active_mat.gain_hh

        rspon = np.zeros((self.omega.size, self.DF.size))
        rspon_broadened = np.zeros((self.omega.size, self.DF.size, 3))
        rspon_lh = np.zeros((self.omega.size, self.DF.size, 3))
        rspon_hh = np.zeros((self.omega.size, self.DF.size, 3))

        for i in range(self.omega.size):
            coeff = (n**2 * self.omega[i]**2) / (pi**2 * hbar * c**2)
            for j in range(self.DF.size):
                if not hbar*self.omega[i] - DF[j] == 0:
                    rspon_lh[i, j, :] = coeff * gain_lh[i, j, :] * 1 / (1 - np.exp(beta*(hbar * self.omega[i] - DF[j])))
                    rspon_hh[i, j, :] = coeff * gain_hh[i, j, :] * 1 / (1 - np.exp(beta*(hbar * self.omega[i] - DF[j])))

        for i in range(self.omega.size):
            for j in range(self.DF.size):
                if hbar*self.omega[i] - DF[j] == 0:
                    rspon_lh[i, j, :] = 0.5 * (rspon_lh[i-1,j,:] + rspon_lh[i+1,j,:])
                    rspon_hh[i, j, :] = 0.5 * (rspon_hh[i-1,j,:] + rspon_hh[i+1,j,:])

        self._rspon_lh = rspon_lh
        self._rspon_hh = rspon_hh

        #rspon = np.sum(rspon_lh, axis = 2) + np.sum(rspon_hh, axis = 2)
        rspon = rspon_lh + rspon_hh
        self._rspon = rspon

    def calc_Rspon(self, rspon):
        rspon_average = np.squeeze(rspon)
        rspon_average = np.sum(rspon_average, axis = 1)

        Rspon = np.trapz(rspon_average, x=hbar*self.omega)
        return Rspon

    def calc_hvRspon(self, rspon):
        rspon_average = np.squeeze(rspon)
        rspon_average = np.sum(rspon_average, axis = 1)

        hvRspon = np.trapz(hbar*self.omega*rspon_average, x=hbar*self.omega)
        return hvRspon

    def calc_Rnr(self, N, P):
        # SRH
        A = self._active_mat.A

        Rsrh = A * (N * P - self._active_mat.ni**2)/(N + P +
                                                    2*self._active_mat.ni)

        #C = 7.0e-42 # Auger coefficient [m^6/s]
        C = self._active_mat.C

        Raug = 0.5*C*(N+P)*(N*P - self._active_mat.ni**2)

        return Rsrh + Raug
        #if self._active_mat.interface_recombination_velocity is not None:
        #    Rirv = 0 # Need to figure out what to do here
        #    return Rsrh + Raug + Rirv
        #else:
        #    return Rsrh + Raug

class nanoLED(LED):
    def __init__(self, active_mat, antenna, srv, sv_ratio):
        super(nanoLED, self).__init__(active_mat)
        self._antenna = antenna

        if self._mat == 'Bulk':
            pass
        elif self._mat == 'QuantumWell':
            self._rspon_lh_ant = None
            self._rspon_hh_ant = None

        self._rspon_ant = None
        self._rspon_ant_broadened = None

        self._srv = srv
        self._sv_ratio = sv_ratio # (2/w + 2/L)

        self._Rnr_ant = None

        self._Rspon_ant = None
        self._hvRspon_ant = None

    @property
    def antenna(self):
        return self._antenna

    @antenna.setter
    def antenna(self, new):
        self._antenna = new
        self._correct_build = False

    @property
    def srv(self):
        return self._srv

    @srv.setter
    def srv(self, val):
        self._srv = val
        self._correct_build = False

    @property
    def sv_ratio(self):
        return self._sv_ratio

    @sv_ratio.setter
    def sv_ratio(self, val):
        self._sv_ratio = val
        self._correct_build = False

    @property
    def rspon_lh_ant(self):
        return self._rspon_lh_ant

    @property
    def rspon_hh_ant(self):
        return self._rspon_hh_ant

    @property
    def rspon_ant(self):
        return self._rspon_ant

    @property
    def rspon_ant_broadened(self):
        return self._rspon_broadened

    @property
    def Rnr_ant(self):
        return self._Rnr

    @property
    def Rspon_ant(self):
        return self._Rspon_ant

    @property
    def hvRspon_ant(self):
        return self._hvRspon_ant

    @property
    def correct_build(self):
        return self._correct_build

    def build(self, broaden_rspon = False, tau_m = 0.1e-12):
        super(nanoLED, self).build(broaden_rspon, tau_m)

        Rspon_ant = np.zeros((self.DF.size))
        hvRspon_ant = np.zeros((self.DF.size))
        Rnr_ant = np.zeros((self.DF.size))
        rad_lifetime_ant = np.zeros((self.DF.size))
        nr_lifetime_ant = np.zeros((self.DF.size))

        self.calc_rspon_ant()
        rspon_ant = self.rspon_ant

        if broaden_rspon:
            rspon_ant_broadened = np.zeros((self.omega.size, self.DF.size, 3))
            for j in range(self.DF.size):
                for n in range(3):
                    rspon_ant_broadened[:,j,n] = \
                        misc.lineshape_broadening(self.omega,rspon_ant[:,j,n], tau_m)
            self._rspon_ant_broadened = rspon_ant_broadened
            rspon_ant = rspon_ant_broadened

        for i in range(self.DF.size):
            Rspon_ant[i] = self.calc_Rspon_ant(rspon_ant[:, i, :])
            hvRspon_ant[i] = self.calc_hvRspon_ant(rspon_ant[:, i, :])
            Rnr_ant[i] = self.calc_Rnr_ant(i, self._active_mat.N[i], self._active_mat.P[i])

        for i in range(1,self.DF.size-1, 1):
            rad_lifetime_ant[i] = self.calc_radiative_lifetime(Rspon_ant, self._active_mat.N, i)
            nr_lifetime_ant[i] = self.calc_radiative_lifetime(Rnr_ant, self._active_mat.N, i)

        self._Rnr_ant = Rnr_ant
        self._Rspon_ant = Rspon_ant
        self._hvRspon_ant = hvRspon_ant
        self._rad_lifetime_ant = rad_lifetime_ant
        self._nr_lifetime_ant = nr_lifetime_ant

    def calc_rspon_ant(self):
        if self._mat == 'Bulk':
            self.calc_rspon_ant_bulk()
        elif self._mat == 'QuantumWell':
            self.calc_rspon_ant_QW()

    def calc_rspon_ant_bulk(self):
        rspon = self.rspon

        Fx = self.antenna.Fx
        Fy = self.antenna.Fy
        Fz = self.antenna.Fz

        rspon_ant = np.zeros((self.omega.size, self.DF.size, 3))

        for i in range(self.DF.size):
            rspon_ant[:,i,0] = rspon[:,i,0] * Fx
            rspon_ant[:,i,1] = rspon[:,i,1] * Fy
            rspon_ant[:,i,2] = rspon[:,i,2] * Fz

        self._rspon_ant = rspon_ant

    def calc_rspon_ant_QW(self):
        rspon_lh = self.rspon_lh
        rspon_hh = self.rspon_hh

        Fx = self.antenna.Fx
        Fy = self.antenna.Fy
        Fz = self.antenna.Fz

        rspon_lh_ant = np.zeros((self.omega.size, self.DF.size, 3))
        rspon_hh_ant = np.zeros((self.omega.size, self.DF.size, 3))

        for i in range(self.DF.size):
            rspon_lh_ant[:,i,0] = rspon_lh[:,i,0] * Fx
            rspon_hh_ant[:,i,0] = rspon_hh[:,i,0] * Fx

            rspon_lh_ant[:,i,1] = rspon_lh[:,i,1] * Fy
            rspon_hh_ant[:,i,1] = rspon_hh[:,i,1] * Fy

            rspon_lh_ant[:,i,2] = rspon_lh[:,i,2] * Fz
            rspon_hh_ant[:,i,2] = rspon_hh[:,i,2] * Fz

        self._rspon_lh_ant = rspon_lh_ant
        self._rspon_hh_ant = rspon_hh_ant

        #rspon_ant = np.sum(rspon_lh_ant, axis = 2) + np.sum(rspon_hh_ant, axis = 2)
        rspon_ant = rspon_lh_ant + rspon_hh_ant
        self._rspon_ant = rspon_ant

    def calc_Rspon_ant(self, rspon_ant):
        rspon_ant_average = np.squeeze(rspon_ant)
        rspon_ant_average = np.sum(rspon_ant_average, axis = 1)

        Rspon_ant = np.trapz(rspon_ant_average, x=hbar*self.omega)
        return Rspon_ant

    def calc_hvRspon_ant(self, rspon_ant):
        rspon_ant_average = np.squeeze(rspon_ant)
        rspon_ant_average = np.sum(rspon_ant_average, axis = 1)

        hvRspon_ant = np.trapz(hbar*self.omega*rspon_ant_average, x=hbar*self.omega)
        return hvRspon_ant

    def calc_Rnr_ant(self, i, N, P):
        Rnr_bare = self.Rnr[i]
        ni = self._active_mat.ni

        if P >= N:
            Rsrv = self.sv_ratio * self.srv * (N*P - ni**2)/P
        else:
            Rsrv = self.sv_ratio * self.srv * (N*P - ni**2)/N

        #Rsrv /= 2.0 # accounts for hole or electron dominated srv?

        return Rnr_bare + Rsrv

    def calc_radiative_lifetime(self, Rspon, N, i):
        dR_dN = (Rspon[i+1] - Rspon[i-1])/(N[i+1]-N[i-1])
        lifetime = dR_dN**(-1)
        return(lifetime)

    def calc_nonradiative_lifetime(self, Rnr, N, i):
        dR_dN = (Rnr[i+1] - Rnr[i-1])/(N[i+1]-N[i-1])
        lifetime = dR_dN**(-1)
        return(lifetime)
        



class Antenna(object):
    def __init__(self, omega, efficiency, Fx, Fy = 1.0, Fz = 1.0):
        self._omega = omega

        self._efficiency = efficiency

        if np.all(Fy == 1.0):
            self._Fy = np.ones(omega.size)
        else:
            self._Fy = Fy

        if np.all(Fz == 1.0):
            self._Fz = np.ones(omega.size)
        else:
            self._Fz = Fz

        self._Fx = Fx # x is TE, y is TE, z is TM for QW

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, val):
        self._omega = val

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, val):
        self._efficiency = val

    @property
    def Fx(self):
        return self._Fx

    @Fx.setter
    def Fx(self, val):
        self._Fx = val

    @property
    def Fy(self):
        return self._Fy

    @Fy.setter
    def Fy(self, val):
        self._Fy = val

    @property
    def Fz(self):
        return self._Fz

    @Fz.setter
    def Fz(self, val):
        self._Fz = val

def purcell_enhancement(omega, w0, Q, Veff):
    # Lorentzian antenna resonance, corresponding to purcell enhance spectrum
    # use this with Antenna class to define enhancement values for each dipole
    # polarization

    F = (3/(4*pi**2)) / Veff * omega * w0 * Q / \
            (4 * Q**2 * (omega - w0)**2 + w0**2)

    return F

def test_nanoLED_QW():
    import matplotlib.pyplot as plt
    import active_material

    n = 3.5
    Egw = 0.755*q
    Ep = 25.7 * q
    Nd = 0.0
    Na = 0.0
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

    omega = np.linspace(0.1, 3.0, num=2000)*q/hbar
    DF_max = 1.1*q
    DF_dis = 1.1*q/500
    T=300.0

    QW = active_material.QuantumWell(omega, DF_max, DF_dis, T, n, mw, mw_lh, mw_hh, Ep, M, Egw,
                     Na, Nd, Lz, Egb, mb, mb_lh, mb_hh, delEc)

    QW.build()
    Fx = np.ones(omega.size)
    antenna = Antenna(omega, Fx)

    nanoLED = nanoLED_QW(omega, QW.DF, T, QW, antenna)
    nanoLED.build()

    f = plt.figure()
    f2 = plt.figure()
    ax = f.add_subplot(111)
    ax2 = f2.add_subplot(111)
    f3 = plt.figure()
    ax3 = f3.add_subplot(111)
    f5 = plt.figure()
    ax5 = f5.add_subplot(111)
    f6 = plt.figure()
    ax6 = f6.add_subplot(111)

    for i in range(0, QW.DF.size, 10):
        ax.plot(h*c/(hbar*omega)*1e6, nanoLED.rspon_ant[:,i])
        ax2.plot(h*c/(hbar*omega)*1e6, nanoLED.rspon_ant_broadened[:,i])
        ax3.plot(h*c/(hbar*omega)*1e6,(nanoLED.rspon_lh_ant[:,i,0]+nanoLED.rspon_hh_ant[:,i,0]))
        ax5.plot(h*c/(hbar*omega)*1e6,(nanoLED.rspon_lh_ant[:,i,1]+nanoLED.rspon_hh_ant[:,i,1]))
        ax6.plot(h*c/(hbar*omega)*1e6,(nanoLED.rspon_lh_ant[:,i,2]+nanoLED.rspon_hh_ant[:,i,2]))

    f4 = plt.figure()
    ax4 = f4.add_subplot(111)
    ax4.loglog(QW.N/1e6, nanoLED.Rspon)


    plt.show()

if __name__ == '__main__':
    test_nanoLED_QW()
