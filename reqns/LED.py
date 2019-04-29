import numpy as np
#!/usr/bin/env python
"""
Provides class LED_bulk, nanoLED_bulk, LED_QW, and nanoLED_QW. Also provides
classes OpticalCavity and Antenna

LED_bulk and LED_QW classes can be used to calculate spontaneous emission
spectra from large volume/area LED devices, as well as radiative and
nonradiative recombination rates. OpticalCavity and Antenna classes allow one
to specify optical antenna enhancement due to the Purcell effect (effectively
modifies the matrix element). nanoLED_bulk and nanoLED_QW classes provide the
spontaneous emission spectra after antenna enhancement.
"""

from math import pi, ceil
from fdint import fdk

from physical_constants import * # h, hbar, c, q, eps0, m0, k (SI)
import misc

__author__ = "Sean Hooten"
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Sean Hooten"
__status__ = "development"

class LED_bulk(object):
    def __init__(self, omega, DF, T, active_mat):
        # inherit this data from active_mat instance?
        self.omega = omega
        self.DF = DF
        self.T = T

        self.active_mat = active_mat
        #self.antenna = antenna

        self.tau = None
        #self.tau_ant = None
        self.rspon = None
        #self.rspon_ant = None

        self.Rnr = None
        self.Rspon = None
        #self.Rspon_ant = None

        self.hvRspon = None
        #self.hvRspon_ant = None

        # Need to make these settable
        #self.vg = c / self.active_mat.n # group velocity
        #self.tau_p = 700 / (self.active_mat.Eg/hbar)
        #self.beta = 0.01
        #self.Gamma = 1


    def build(self, update_flag=False):
        if update_flag:
            self.active_mat.update(self.omega, self.DF, self.T)

        self.unenhanced_lifetime()

        rspon = np.zeros((self.omega.size, self.DF.size))

        Rspon = np.zeros((self.DF.size))
        hvRspon = np.zeros((self.DF.size))

        Rnr = np.zeros((self.DF.size))

        for i in range(self.DF.size):
            rspon[:, i] = self.calc_rspon(self.active_mat.fe[:, i])
            Rspon[i] = self.calc_Rspon(rspon[:, i])
            Rnr[i] = self.calc_Rnr(self.active_mat.N[i], self.active_mat.P[i])

            hvRspon[i] = self.calc_hvRspon(rspon[:, i])

        self.rspon = rspon
        self.Rspon = Rspon
        self.Rnr = Rnr
        self.hvRspon = hvRspon

    def update(self, omega, DF, T, active_mat, update_flag=False):
        self.omega = omega
        self.DF = DF
        self.T = T

        self.active_mat = active_mat

        self.build(update_flag)

    def calc_photonDOS(self):
        DOSphot = self.active_mat.n**2 * self.omega**2 / (hbar * pi**2 * c**3)
        return DOSphot

    def unenhanced_lifetime(self):
        n = self.active_mat.n
        M = self.active_mat.M

        tau = (n**2 * self.omega**2) / (hbar * pi**2 * c**2) * pi * q**2 / \
                (n * c * eps0 * m0**2 * self.omega) * M
        tau = 1/tau
        self.tau = tau

    def calc_rspon(self, fe):
        rspon = (1/self.tau) * self.active_mat.rho * fe
        return rspon

    def calc_Rspon(self, rspon):
        Rspon = np.trapz(rspon, x=hbar*self.omega)
        return Rspon

    def calc_hvRspon(self, rspon):
        hvRspon = np.trapz(hbar*self.omega*rspon, x=hbar*self.omega)
        return hvRspon

    def calc_Rnr(self, N, P):
        # SRH
        A = (1/(5.0e-8)) #SRH coefficient [1/s]
        Rsrh = A * (N * P - self.active_mat.ni**2)/(N + P +
                                                    2*self.active_mat.ni)



        #C = 7.0e-42 # Auger coefficient [m^6/s]
        C = 1e-50
        Raug = 0.5*C*(N+P)*(N*P - self.active_mat.ni**2)

        return Rsrh + Raug

class nanoLED_bulk(LED_bulk):
    def __init__(self, omega, DF, T, active_mat, antenna):
        # inherit this data from active_mat instance?
        super(nanoLED_bulk, self).__init__(omega, DF, T, active_mat)

        self.antenna = antenna

        self.tau_ant = None
        self.rspon_ant = None

        #self.Rnr = None (??????)
        self.Rspon_ant = None
        self.hvRspon_ant = None

        # Need to make these settable
        #self.vg = c / self.active_mat.n # group velocity
        #self.tau_p = 700 / (self.active_mat.Eg/hbar)
        #self.beta = 0.01
        #self.Gamma = 1


    def build(self, update_flag=False, F=np.array([])):
        if update_flag:
            self.active_mat.update(self.omega, self.DF, self.T)
            super(nanoLED_bulk, self).build(update_flag)
            if F.size == 0:
                self.antenna.update(self.omega)
            else:
                self.antenna.update(self.omega, F)

        self.lifetime()

        rspon_ant = np.zeros((self.omega.size, self.DF.size))

        Rspon_ant = np.zeros((self.DF.size))
        hvRspon_ant = np.zeros((self.DF.size))

        Rnr = np.zeros((self.DF.size))

        for i in range(self.DF.size):
            rspon_ant[:, i] = self.calc_rspon_ant(self.active_mat.fe[:, i])
            Rspon_ant[i] = self.calc_Rspon_ant(rspon_ant[:, i])
            #Rnr[i] = self.calc_Rnr(self.active_mat.N[i], self.active_mat.P[i])

            hvRspon_ant[i] = self.calc_hvRspon_ant(rspon_ant[:, i])

        self.rspon_ant = rspon_ant
        self.Rspon_ant = Rspon_ant
        #self.Rnr = Rnr
        self.hvRspon_ant = hvRspon_ant

    def update(self, omega, DF, T, active_mat, antenna, update_flag=False,
               F=np.array([])):
        self.omega = omega
        self.DF = DF
        self.T = T

        self.active_mat = active_mat
        self.antenna = antenna

        self.build(update_flag, F)


    def lifetime(self):
        tau_ant = self.tau / self.antenna.F
        self.tau_ant = tau_ant

    def calc_rspon_ant(self, fe):
        rspon_ant = (1/self.tau_ant) * self.active_mat.rho * fe
        return rspon_ant

    def calc_Rspon_ant(self, rspon_ant):
        Rspon_ant = np.trapz(rspon_ant, x=hbar*self.omega)
        return Rspon_ant

    def calc_hvRspon_ant(self, rspon_ant):
        hvRspon_ant = np.trapz(hbar*self.omega*rspon_ant, x=hbar*self.omega)
        return hvRspon_ant

    #def calc_Rnr(self, N, P):
    #    # SRH
    #    A = (1/(5.0e-8)) #SRH coefficient [1/s]
    #    Rsrh = A * (N * P - self.active_mat.ni**2)/(N + P +
    #                                                2*self.active_mat.ni)



    #    #C = 7.0e-42 # Auger coefficient [m^6/s]
    #    C = 1e-50
    #    Raug = 0.5*C*(N+P)*(N*P - self.active_mat.ni**2)

    #    return Rsrh + Raug



class LED_QW(object):
    def __init__(self, active_mat):
        self.active_mat = active_mat

        self.omega = active_mat.omega
        self.DF = active_mat.DF
        self.T = active_mat.T

        # How to model this?
        self.rspon = None
        self.rspon_broadened = None
        self.rspon_lh = None
        self.rspon_hh = None

        self.Rnr = None
        self.Rspon = None

        self.hvRspon = None

    def build(self, update_flag=False):
        if update_flag:
            self.active_mat.update(self.omega, self.DF, self.T)

        Rspon = np.zeros((self.DF.size))
        hvRspon = np.zeros((self.DF.size))

        Rnr = np.zeros((self.DF.size))

        self.calc_rspon()
        #rspon = self.rspon
        rspon = self.rspon_broadened

        for i in range(self.DF.size):
            Rspon[i] = self.calc_Rspon(rspon[:, i, :])
            Rnr[i] = self.calc_Rnr(self.active_mat.N[i], self.active_mat.P[i])

            hvRspon[i] = self.calc_hvRspon(rspon[:, i, :])

        self.Rspon = Rspon
        self.Rnr = Rnr
        self.hvRspon = hvRspon

    def update(self, omega, DF, T, active_mat, update_flag=False):
        self.omega = omega
        self.DF = DF
        self.T = T

        self.active_mat = active_mat

        self.build(update_flag)

    def calc_photonDOS(self):
        DOSphot = self.active_mat.n**2 * self.omega**2 / (hbar * pi**2 * c**3)
        return DOSphot

    def calc_rspon(self):
        DF = self.DF
        n = self.active_mat.n
        beta = (1 / (k * self.T))

        gain_lh = self.active_mat.gain_lh
        gain_hh = self.active_mat.gain_hh

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

        self.rspon_lh = rspon_lh
        self.rspon_hh = rspon_hh

        #rspon = np.sum(rspon_lh, axis = 2) + np.sum(rspon_hh, axis = 2)
        rspon = rspon_lh + rspon_hh
        self.rspon = rspon

        # lineshape broadening
        for j in range(self.DF.size):
            for n in range(3):
                rspon_broadened[:,j,n] = misc.lineshape_broadening(self.omega,rspon[:,j,n], 0.1e-12)

        self.rspon_broadened = rspon_broadened

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
        A = self.active_mat.A

        Rsrh = A * (N * P - self.active_mat.ni**2)/(N + P +
                                                    2*self.active_mat.ni)

        #C = 7.0e-42 # Auger coefficient [m^6/s]
        C = self.active_mat.C

        Raug = 0.5*C*(N+P)*(N*P - self.active_mat.ni**2)

        if self.active_mat.interface_recombination_velocity is not None:
            Rirv = 0 # Need to figure out what to do here
            return Rsrh + Raug + Rirv
        else:
            return Rsrh + Raug


class nanoLED_QW(LED_QW):
    def __init__(self, active_mat, antenna, vs, srv_surface):
        # inherit this data from active_mat instance?
        super(nanoLED_QW, self).__init__(active_mat)

        self.antenna = antenna

        #self.tau_ant = None
        self.rspon_ant = None
        self.rspon_ant_broadened = None
        self.rspon_lh_ant = None
        self.rspon_hh_ant = None

        self.srv_surface = srv_surface # (2/w + 2/L)
        self.vs = vs

        self.Rnr_ant = None

        self.Rspon_ant = None
        self.hvRspon_ant = None



    def build(self, update_flag=False, F=np.array([])):
        #if update_flag:
        #    self.active_mat.update(self.omega, self.DF, self.T)
        #    super(nanoLED_QW, self).build(update_flag)
        #    if F.size == 0:
        #        self.antenna.update(self.omega)
        #    else:
        #        self.antenna.update(self.omega, F)

        super(nanoLED_QW, self).build(update_flag)

        Rspon_ant = np.zeros((self.DF.size))
        hvRspon_ant = np.zeros((self.DF.size))

        Rnr_ant = np.zeros((self.DF.size))

        self.calc_rspon_ant()
        rspon_ant = self.rspon_ant_broadened

        for i in range(self.DF.size):
            Rspon_ant[i] = self.calc_Rspon_ant(rspon_ant[:, i, :])
            Rnr_ant[i] = self.calc_Rnr_ant(i, self.active_mat.N[i], self.active_mat.P[i])

            hvRspon_ant[i] = self.calc_hvRspon_ant(rspon_ant[:, i, :])

        self.Rnr_ant = Rnr_ant
        self.Rspon_ant = Rspon_ant
        self.hvRspon_ant = hvRspon_ant

        #import matplotlib.pyplot as plt
        #f = plt.figure()
        #f2 = plt.figure()
        #ax = f.add_subplot(111)
        #ax2 = f2.add_subplot(111)
        #ax.plot(2*pi*c/self.omega*1e6, self.rspon_ant_broadened)
        #ax2.semilogy(2*pi*c/self.omega*1e6, self.rspon_ant_broadened)

        #plt.show()

    def update(self, omega, DF, T, active_mat, antenna, update_flag=False,
               F=np.array([])):
        self.omega = omega
        self.DF = DF
        self.T = T

        self.active_mat = active_mat
        self.antenna = antenna

        self.build(update_flag, F)


    def calc_rspon_ant(self):
        rspon_lh = self.rspon_lh
        rspon_hh = self.rspon_hh

        Fx = self.antenna.Fx
        Fy = self.antenna.Fy
        Fz = self.antenna.Fz

        rspon_lh_ant = np.zeros((self.omega.size, self.DF.size, 3))
        rspon_hh_ant = np.zeros((self.omega.size, self.DF.size, 3))

        for i in range(self.omega.size):
            rspon_lh_ant[i,:,0] = rspon_lh[i,:,0] * Fx[i]
            rspon_hh_ant[i,:,0] = rspon_hh[i,:,0] * Fx[i]

            rspon_lh_ant[i,:,1] = rspon_lh[i,:,1] * Fy[i]
            rspon_hh_ant[i,:,1] = rspon_hh[i,:,1] * Fy[i]

            rspon_lh_ant[i,:,2] = rspon_lh[i,:,2] * Fz[i]
            rspon_hh_ant[i,:,2] = rspon_hh[i,:,2] * Fz[i]

        self.rspon_lh_ant = rspon_lh_ant
        self.rspon_hh_ant = rspon_hh_ant

        #rspon_ant = np.sum(rspon_lh_ant, axis = 2) + np.sum(rspon_hh_ant, axis = 2)
        rspon_ant = rspon_lh_ant + rspon_hh_ant
        self.rspon_ant = rspon_ant
        
        #lineshape broadening
        rspon_ant_broadened = np.zeros((self.omega.size, self.DF.size, 3))

        for j in range(self.DF.size):
            for n in range(3):
                rspon_ant_broadened[:,j,n] = misc.lineshape_broadening(self.omega,
                                                                 rspon_ant[:,j,n],
                                                                 0.1e-12)

        self.rspon_ant_broadened = rspon_ant_broadened
        #rspon_ant = (1/self.tau_ant) * self.active_mat.rho * fe
        #return rspon_ant

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
        ni = self.active_mat.ni

        if P >= N:
            Rsrv = self.srv_surface * self.vs * (N*P - ni**2)/P
        else:
            Rsrv = self.srv_surface * self.vs * (N*P - ni**2)/N

        #Rsrv /= 2.0 # accounts for hole or electron dominated srv?

        return Rnr_bare + Rsrv

class OpticalCavity(object):
    # Lorentzian antenna resonance, corresponding to purcell enhance spectrum
    def __init__(self, omega, w0, Q, Veff):
        self.omega = omega
        self.w0 = w0
        self.Q = Q
        self.Veff = Veff # normalized by lambda^3

        self.F = None
        self.calc_F()

    def calc_F(self):
        F = (3/(4*pi**2)) / self.Veff * self.omega * self.w0 * self.Q / \
                (4 * self.Q**2 * (self.omega - self.w0)**2 + self.w0**2)
        self.F = F

    def update(self, omega):
        self.omega = omega
        self.calc_F()

class Antenna(object):
    def __init__(self, omega, Fx, Fy = 1.0, Fz = 1.0):
        self.omega = omega

        if np.all(Fy == 1.0):
            self.Fy = np.ones(omega.size)
        else:
            self.Fy = Fy

        if np.all(Fz == 1.0):
            self.Fz = np.ones(omega.size)
        else:
            self.Fz = Fz

        self.Fx = Fx # x is TE, y is TE, z is TM for QW

    def update(self, omega, Fx=np.array([]), Fy = 1, Fz = 1):
        self.omega = omega
        if Fx.size == 0:
            print('Warning, antenna F needs to be updated explicitly')
        else:
            self.Fx = Fx

        if Fy == 1:
            self.Fy = np.ones(omega.size)
        else:
            self.Fy = Fy

        if Fz == 1:
            self.Fz = np.ones(omega.size)
        else:
            self.Fz = Fz

if __name__ == '__main__':
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



    import matplotlib.pyplot as plt
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
