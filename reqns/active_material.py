import numpy as np
from math import pi, ceil
from physical_constants import * # h, hbar, c, q, eps0, m0, k (SI)
from scipy.optimize import fsolve
import misc

class ActiveMaterial(object):
    def __init__(self):
        pass

    def fermi_dirac_function(self, E, F, T):
        kT = k*T
        f = 1 / (1 + np.exp( (E - F) / kT ))
        return f

    def fermi_inversion_function(self, E1, Fv, E2, Fc, T):
        f2 = self.fermi_dirac_function(E2, Fc, T)
        f1 = self.fermi_dirac_function(E1, Fv, T)
        return f2 - f1

    def fermi_emission_factor(self, E1, Fv, E2, Fc, T):
        f1 = 1 - self.fermi_dirac_function(E1, Fv, T)
        f2 = self.fermi_dirac_function(E2, Fc, T)
        return f2 * f1

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

        self.n0 = n0
        self.p0 = p0

class Bulk(ActiveMaterial):
    def __init__(self, omega, DF_max, DF_dis, T, n, me, mh, Ep, M, Eg, ni, Nc,
                Nv, Na, Nd):
        # Required Material Data
        self.n = n
        self.me = me
        self.mh = mh
        self.mr = 1 / (1/self.me + 1/self.mh)
        self.Ep = Ep
        self.M = M
        self.Eg = Eg
        self.ni = ni
        self.Nc = Nc
        self.Nv = Nv

        self.Na = Na
        self.Nd = Nd
        
        # Required User Inputs
        self.DF_max = DF_max # assume minimum 0, max defined by source
        self.T = T # material data currently only valid for T = 300K
        self.omega = omega #must choose sufficiently discretized and large
        self.DF_dis = DF_dis # (roughly) discretization of DF points

        # To be calculated with self.build()
        self.DF = None
        self.n0 = None
        self.p0 = None
        self.Fc = None
        self.Fv = None
        self.rho = None
        self.gain = None
        self.N = None
        self.P = None
        self.E1 = None
        self.E2 = None

        self.fg = None
        self.fe = None

    def build(self):
        self.calc_n0_p0()
        self.calc_N_P()
        self.calc_DOSjoint()
        self.calc_E1()
        self.calc_E2()

        gain = np.zeros((self.omega.size, self.DF.size))
        fe = np.zeros((self.omega.size, self.DF.size))
        fg = np.zeros((self.omega.size, self.DF.size))

        for i in range(self.DF.size):
            fg[:, i] = self.fermi_inversion_function(self.Fv[i], self.Fc[i], self.T)
            fe[:, i] = self.fermi_emission_factor(self.Fv[i], self.Fc[i], self.T)
            gain[:, i] = self.get_gain(fg[:, i])

        self.fg = fg
        self.fe = fe
        self.gain = gain


    def update(self, omega, DF_max, DF_dis, T):
        # Material data can be updated simply by setting the material values
        self.DF_max = DF_max
        self.DF_dis = DF_dis
        self.T = T
        self.omega = omega

        self.build()

    def fermi_inversion_function(self, Fv, Fc, T):
        E1 = self.E1
        E2 = self.E2

        fg = super(Bulk, self).fermi_inversion_function(E1, Fv, E2, Fc, T)

        return fg

    def fermi_emission_factor(self, Fv, Fc, T):
        E1 = self.E1
        E2 = self.E2

        fe = super(Bulk, self).fermi_emission_factor(E1, Fv, E2, Fc, T)

        return fe

    def get_gain(self, fg):
        rho = self.rho
        C0 = pi*q**2 / (self.n * c * eps0 * m0**2 * self.omega)
        gain = C0 * rho * self.M * fg

        return gain

    def calc_N_P(self):
        num = int(ceil((self.Eg+self.DF_max) / self.DF_dis))
        Fcs = np.linspace(0, self.Eg+self.DF_max, num=num)
        Fvs = np.linspace(-self.DF_max, self.Eg, num=num)

        beta = 1/(k*self.T)
        N = np.zeros(num)
        P = np.zeros(num)

        N = self.Nc * misc.fermi_dirac_integral(k=0.5, phi=beta*(Fcs-self.Eg))
        P = self.Nv * misc.fermi_dirac_integral(k=0.5, phi=beta*(-Fvs))

        #N = self.Nc * np.exp(beta*(Fcs-self.Eg))
        #P = self.Nv * np.exp(beta*(-Fvs))

        # when Fc = Fv = Fref, n=n0, p=p0
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

        self.N = NN
        self.P = PP
        self.DF = DF

        self.Fc = Fc
        self.Fv = Fv

        #import matplotlib.pyplot as plt
        #plt.rcParams.update({'font.size':20})
        #f = plt.figure()
        #ax = f.add_subplot(111)
        #ax.semilogy(DF/q, NN/1e6, '-o')
        #ax.semilogy(DF/q, PP/1e6, '-o')
        #plt.xlabel(r'$\Delta F$ (V)')
        #plt.ylabel(r'N, P (cm$^{-3}$)')
        #plt.show()

    def calc_DOSjoint(self):
        args = hbar*self.omega - self.Eg
        arg = np.array([x if x>=0 else 0 for x in args]) # test for negatives
        rho = (1 / (2*pi**2)) * (2*self.mr/(hbar**2))**(1.5) * np.sqrt(arg)

        ######## WHAT IS THIS? ########
        rho = 4*rho # factor of 4 for spin degeneracy? Exciton effects?
        ###############################

        self.rho = rho

    def calc_E1(self):
        E1 = -1 * (hbar*self.omega - self.Eg) * self.mr / self.mh
        self.E1 = E1

    def calc_E2(self):
        E2 = self.Eg + (hbar*self.omega - self.Eg) * self.mr / self.me
        self.E2 = E2


class QuantumWell(ActiveMaterial):
    def __init__(self, omega, DF_max, DF_dis, T, n, mw, mw_lh, mw_hh, Ep, M,
                 Egw, Na, Nd, Lz, Egb, mb, mb_lh, mb_hh, delEc, A, C):
        # Required Material Data
        # MAKE SURE ALL VALUES ARE SI

        self.n = n
        self.Ep = Ep
        self.M = M

        self.Na = Na
        self.Nd = Nd

        self.Egw = Egw # quantum well bandgap
        self.Egb = Egb # barrier bandgap
        self.Lz = Lz # quantum well depth
        self.mw = mw
        self.mw_hh = mw_hh
        self.mw_lh = mw_lh
        self.mb = mb
        self.mb_hh = mb_hh
        self.mb_lh = mb_lh

        self.delEc = delEc

        self.mr_hh = mw_hh*mw/(mw_hh + mw)
        self.mr_lh = mw_lh*mw/(mw_lh + mw)

        self.Nc = mw * k * T / (pi * hbar**2 * Lz)
        self.Nv_hh = mw_hh * k * T / (pi * hbar**2 * Lz)
        self.Nv_lh = mw_lh * k * T / (pi * hbar**2 * Lz)

        self.A = A
        self.C = C

        self.interface_recombination_velocity = None

        # Required User Inputs
        self.DF_max = DF_max # assume minimum 0, max defined by source
        self.T = T # material data currently only valid for T = 300K
        self.omega = omega #must choose sufficiently discretized and large
        self.DF_dis = DF_dis # (roughly) discretization of DF points

        # To be calculated with self.build()
        self.ni = None
        self.DF = None
        self.n0 = None
        self.p0 = None
        self.Fc = None
        self.Fv = None

        self.rho_lh = None
        self.rho_hh = None
        self.gain = None
        self.gain_broadened = None
        self.N = None
        self.P = None

        self.E1_lh = None
        self.E2_lh = None

        self.E1_hh = None
        self.E2_hh = None

        self.fg_hh = None
        self.fg_lh = None

        self.Ew_e = None
        self.Ew_lh = None
        self.Ew_hh = None

    def build(self):
        self.calc_eigen_energies_finite()
        self.calc_N_P()
        self.calc_DOSjoint()
        self.calc_E1()
        self.calc_E2()

        fe_lh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))
        fg_lh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))
        fe_hh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))
        fg_hh = np.zeros((self.omega.size, self.DF.size, self.Ew_e.size))

        gain_broadened = np.zeros((self.omega.size, self.DF.size))

        for i in range(self.DF.size):
            for j in range(self.Ew_e.size):
                for n in range(self.omega.size):
                    fg_lh[n, i, j] = self.fermi_inversion_function(self.E1_lh[n,j], self.Fv[i],
                                                               self.E2_lh[n,j], self.Fc[i], self.T)
                    fg_hh[n, i, j] = self.fermi_inversion_function(self.E1_hh[n,j], self.Fv[i],
                                                               self.E2_hh[n,j], self.Fc[i], self.T)
                    fe_lh[n, i, j] = self.fermi_emission_factor(self.E1_lh[n,j],self.Fv[i],
                                                                self.E2_lh[n,j], self.Fc[i], self.T)
                    fe_hh[n, i, j] = self.fermi_emission_factor(self.E1_hh[n,j],self.Fv[i],
                                                               self.E2_hh[n,j], self.Fc[i], self.T)


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

        self.fg_lh = fg_lh
        self.fg_hh = fg_hh
        self.fe_lh = fe_lh
        self.fe_hh = fe_hh

        self.calc_gain2()

        #for i in range(self.DF.size):
        #    gain_broadened[:,i] = misc.lineshape_broadening(self.omega,self.gain[:,i],0.1e-12)

        #self.gain_broadened = gain_broadened

    def update(self, omega, DF_max, DF_dis, T):
        # Material data can be updated simply by setting the material values
        self.DF_max = DF_max
        self.DF_dis = DF_dis
        self.T = T
        self.omega = omega

        self.build()

    def calc_gain(self):
        rho_lh = self.rho_lh
        rho_hh = self.rho_hh

        fg_lh = self.fg_lh
        fg_hh = self.fg_hh

        C0 = pi*q**2 / (self.n * c * eps0 * m0**2 * self.omega)

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
                            gain_lh[i,n] += C0[i] * self.M * rho_lh * fg_lh[i,n,j]


                        if hbar*self.omega[i] >= self.Ew_e[j] + self.Ew_hh[j] + self.Egw:
                            gain_hh[i,n] += C0[i] * self.M * rho_hh * fg_hh[i,n,j]
                else:
                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_lh[0] + self.Egw:
                        gain_lh[i,n] += C0[i] * self.M * rho_lh * fg_lh[i,n,0]

                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_hh[0] + self.Egw:
                        gain_hh[i,n] += C0[i] * self.M * rho_hh * fg_hh[i,n,0]

        # Are these supposed to be normalized?
        gain_lh_pol[:,:,0] = gain_lh * (1.0/2.0)
        gain_lh_pol[:,:,1] = gain_lh * (1.0/2.0)
        gain_lh_pol[:,:,2] = gain_lh * 2.0

        gain_hh_pol[:,:,0] = gain_hh * (3.0/2.0)
        gain_hh_pol[:,:,1] = gain_hh * (3.0/2.0)
        gain_hh_pol[:,:,2] = gain_hh * 0.0

        self.gain_lh = gain_lh_pol
        self.gain_hh = gain_hh_pol
        self.gain = 3*gain_lh + 3*gain_hh # times 3 for contributions of all polarizations?

    def calc_gain2(self):
        rho_lh = self.rho_lh
        rho_hh = self.rho_hh

        fg_lh = self.fg_lh
        fg_hh = self.fg_hh

        C0 = pi*q**2 / (self.n * c * eps0 * m0**2 * self.omega)

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
                            gain_lh_pol[i,n,0] += C0[i] * rho_lh * fg_lh[i,n,j] * self.M * (5.0/4.0-3.0/4.0*cos2)
                            gain_lh_pol[i,n,1] += C0[i] * rho_lh * fg_lh[i,n,j] * self.M * (5.0/4.0-3.0/4.0*cos2)
                            gain_lh_pol[i,n,2] += C0[i] * rho_lh * fg_lh[i,n,j] * self.M * (1.0/2.0+3.0/2.0*cos2)

                        if hbar*self.omega[i] >= self.Ew_e[j] + self.Ew_hh[j] + self.Egw:
                            cos2 = (self.Ew_e[j]+self.Ew_hh[j])/(hbar*self.omega[i]-self.Egw)
                            gain_hh_pol[i,n,0] += C0[i] * rho_hh * fg_hh[i,n,j] * self.M * 3.0/4.0 * (1+cos2)
                            gain_hh_pol[i,n,1] += C0[i] * rho_hh * fg_hh[i,n,j] * self.M * 3.0/4.0 * (1+cos2)
                            gain_hh_pol[i,n,2] += C0[i] * rho_hh * fg_hh[i,n,j] * self.M * 3.0/2.0 * (1-cos2)
                else:

                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_lh[0] + self.Egw:
                        cos2 = (self.Ew_e[0]+self.Ew_lh[0])/(hbar*self.omega[i]-self.Egw)
                        gain_lh_pol[i,n,0] += C0[i] * rho_lh * fg_lh[i,n,0]  * self.M * (5.0/4.0-3.0/4.0*cos2)
                        gain_lh_pol[i,n,1] += C0[i] * rho_lh * fg_lh[i,n,0]  * self.M * (5.0/4.0-3.0/4.0*cos2)
                        gain_lh_pol[i,n,2] += C0[i] * rho_lh * fg_lh[i,n,0]  * self.M * (1.0/2.0+3.0/2.0*cos2)

                    if hbar*self.omega[i] >= self.Ew_e[0] + self.Ew_hh[0] + self.Egw:
                        cos2 = (self.Ew_e[0]+self.Ew_hh[0])/(hbar*self.omega[i]-self.Egw)
                        gain_hh_pol[i,n,0] += C0[i] * rho_hh * fg_hh[i,n,0] * self.M * 3.0/4.0 * (1+cos2)
                        gain_hh_pol[i,n,1] += C0[i] * rho_hh * fg_hh[i,n,0] * self.M * 3.0/4.0 * (1+cos2)
                        gain_hh_pol[i,n,2] += C0[i] * rho_hh * fg_hh[i,n,0] * self.M * 3.0/2.0 * (1-cos2)

        self.gain_lh = gain_lh_pol
        self.gain_hh = gain_hh_pol
        self.gain = np.sum(gain_lh_pol, axis=2) + np.sum(gain_hh_pol, axis=2)

    def calc_N_P(self):
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
        self.ni = N[ref_ni] # check this

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

        self.N = NN
        self.P = PP
        self.DF = DF

        self.Fc = Fc
        self.Fv = Fv

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

    def calc_eigen_energies(self):
        #Vo_e = (self.Egb - self.Egw) * delEc / q # this is a voltage
        #Vo_h = (self.Egb - self.Egw) / q  - Vo_e

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

        self.Ew_e = np.array(Ew_e)
        self.Ew_lh = np.array(Ew_lh)
        self.Ew_hh = np.array(Ew_hh)

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

        self.Ew_e = np.array(Ew_e)
        self.Ew_lh = np.array(Ew_lh)
        self.Ew_hh = np.array(Ew_hh)

        print self.Ew_e / q
        print self.Ew_lh / q
        print self.Ew_hh / q

    def calc_DOSjoint(self):
        self.rho_hh = self.mr_hh / (pi * hbar**2 * self.Lz)
        self.rho_lh = self.mr_lh / (pi * hbar**2 * self.Lz)
        # THIS IS NOT A SPECTRUM

    def calc_E1(self):
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
        self.E1_hh = E1_hh
        self.E1_lh = E1_lh

    def calc_E2(self):
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
        self.E2_hh = E2_hh
        self.E2_lh = E2_lh

if __name__ == "__main__":
    n = 3.5
    Egw = 0.755*q
    Ep = 25.7 * q
    Nd = 0.0
    Na = 1.0e16*1e6
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
    DF_max = 2.5*q
    DF_dis = 2.5*q/1000
    T=77.0

    QW = QuantumWell(omega, DF_max, DF_dis, T, n, mw, mw_lh, mw_hh, Ep, M, Egw,
                     Na, Nd, Lz, Egb, mb, mb_lh, mb_hh, delEc)

    QW.build()
    print QW.gain_lh.shape
    print QW.gain_hh.shape
    import matplotlib.pyplot as plt
    f = plt.figure()
    f2 = plt.figure()
    f3 = plt.figure()
    f4 = plt.figure()
    f5 = plt.figure()
    f6 = plt.figure()
    f7 = plt.figure()
    f8 = plt.figure()
    ax = f.add_subplot(111)
    ax2 = f2.add_subplot(111)
    ax3 = f3.add_subplot(111)
    ax4 = f4.add_subplot(111)
    ax5 = f5.add_subplot(111)
    ax6 = f6.add_subplot(111)
    ax7 = f7.add_subplot(111)
    ax8 = f8.add_subplot(111)

    for i in range(0, QW.DF.size, 10):
        ax.plot(hbar*omega/q, QW.gain_broadened[:,i])
        ax2.plot(hbar*omega/q, QW.gain[:,i])
        ax3.plot(hbar*omega/q, QW.gain_lh[:,i,0])
        ax4.plot(hbar*omega/q, QW.gain_hh[:,i,0])
        ax5.plot(hbar*omega/q, QW.gain_lh[:,i,1])
        ax6.plot(hbar*omega/q, QW.gain_hh[:,i,1])
        ax7.plot(hbar*omega/q, QW.gain_lh[:,i,2])
        ax8.plot(hbar*omega/q, QW.gain_hh[:,i,2])
    plt.show()
