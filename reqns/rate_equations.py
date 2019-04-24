import numpy as np
from math import pi, ceil
from physical_constants import * # h, hbar, c, q, eps0, m0, k (SI)
#import material_config
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
#from fdint import fdk
import misc

class RateEquations(object):
    def __init__(self, LED, t, I_func, V=1,
                 N0=[0.0], etai=1, max_step=1e-12):
        self.LED = LED
        self.I_func = I_func
        #self.__I_func = I_func
        self.etai = etai
        self.t = t
        self.V = V
        self.max_step = max_step
        self.N0 = N0

        # Need to specify which is minority carrier?


    @property
    def I_func(self):
        return self.__I_func

    @I_func.setter
    def I_func(self, I_func):
        self.__I_func = I_func


    def update(self, t, I_func, V=1,
               N0=[0.0], etai=1, max_step=1e-12):
        # LED updated separately
        self.t = t
        self.I_func = I_func
        self.etai = etai
        self.V = V
        self.max_step = max_step
        self.N0 = N0

    def get_Rspon(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)
        Rspon = np.squeeze(self.LED.Rspon)

        Rspon_int = np.interp(Nq, N, Rspon)
        return Rspon_int

    def get_Rspon_ant(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)
        Rspon_ant = np.squeeze(self.LED.Rspon_ant) # FIX THIS

        Rspon_ant_int = np.interp(Nq, N, Rspon_ant)
        return Rspon_ant_int


    def get_Rnr(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)

        Rnr = np.squeeze(self.LED.Rnr)
        Rnr_int = np.interp(Nq, N, Rnr)
        return Rnr_int

    def get_Rnr_ant(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)

        Rnr_ant = np.squeeze(self.LED.Rnr_ant)
        Rnr_int = np.interp(Nq, N, Rnr_ant)
        return Rnr_int

    def run(self, option='nanoLED'):
        if option == 'nanoLED':
            Nt = solve_ivp(self.compute_RHS_ant, self.t, self.N0, max_step=self.max_step)
        elif option == 'LED':
            Nt = solve_ivp(self.compute_RHS, self.t, self.N0, max_step=self.max_step)

        return np.squeeze(Nt.t), np.squeeze(Nt.y)

    def compute_RHS(self, t, y):
        RHS = self.etai/(q*self.V) * self.I_func(t) - self.get_Rspon(y) \
                    - self.get_Rnr(y)

        return RHS

    def compute_RHS_ant(self, t, y):
        RHS = self.etai/(q*self.V) * self.I_func(t) - self.get_Rspon_ant(y) \
                    - self.get_Rnr_ant(y)
        return RHS

    def calc_peak_to_peak_power_fit(self, t, Nt, w, t_start):
        Nt = np.array(Nt)
        t = np.array(t)

        Nt = Nt[np.where(t>=t_start)]
        t = t[np.where(t>=t_start)]
        p0 = [0.01*Nt[0], 1, Nt[0]]

        params, _ = curve_fit(misc.sinusoid, w*t, Nt, p0=p0)

        max_Nt = params[2]+np.abs(params[0])
        min_Nt = params[2]-np.abs(params[0])

        max_hvRspon = self.get_hvRspon(max_Nt)
        min_hvRspon = self.get_hvRspon(min_Nt)

        deltaP = max_hvRspon - min_hvRspon

        return deltaP

    def calc_peak_to_peak_power(self, t, Nt, t_start, option):
        Nt = np.array(Nt)
        t = np.array(t)

        Nt = Nt[np.where(t>=t_start)]
        t = t[np.where(t>=t_start)]

        max_Nt = np.amax(Nt)
        min_Nt = np.amin(Nt)
        if option == 'LED':
            max_hvRspon = self.get_hvRspon(max_Nt)
            min_hvRspon = self.get_hvRspon(min_Nt)
        elif option == 'nanoLED':
            max_hvRspon = self.get_hvRspon_ant(max_Nt)
            min_hvRspon = self.get_hvRspon_ant(min_Nt)

        deltaP = max_hvRspon - min_hvRspon

        return deltaP

    def calc_dc_power(self, t, Nt, t_start, option):
        Nt = np.array(Nt)
        t = np.array(t)

        Nt = Nt[np.where(t>=t_start)]
        t = t[np.where(t>=t_start)]


        if option == 'LED':
            hvRspon = self.get_hvRspon(Nt)
            power_avg = (1/(t[-1]-t[0])) * np.trapz(hvRspon, x=t)

        elif option == 'nanoLED':
            hvRspon = self.get_hvRspon_ant(Nt)
            power_avg = (1/(t[-1]-t[0])) * np.trapz(hvRspon, x=t)

        return power_avg


    def get_hvRspon(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)

        hvRspon = np.squeeze(self.LED.hvRspon)

        hvRspon_int = np.interp(Nq, N, hvRspon)
        return hvRspon_int

    def get_hvRspon_ant(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)

        hvRspon_ant = np.squeeze(self.LED.hvRspon_ant)

        hvRspon_ant_int = np.interp(Nq, N, hvRspon_ant)
        return hvRspon_ant_int

    def find_desired_Idc(self, t, max_step, n, I_guess, option='nanoLED',
                         tol=1.0e-4, t_start=0.95):
        # currently n is a desired n to find (support for multiple n's later)
        # I_guess is a 2 element array that defines binary search
        # Remember to update reqns object after using this method

        n = np.array([n])
        found_list = [False for i in n]
        I_list = np.zeros((n.size))

        self.t = t
        self.max_step = max_step

        print 'got here'
        self.I_func = lambda t: I_guess[0]
        tt_low, n_low = self.run(option=option)
        #n_low = n_low[-1]

        self.I_func = lambda t: I_guess[1]
        tt_high, n_high = self.run(option=option)
        #n_high = n_high[-1]
        #print n_low, n_high

        n_low = n_low[np.where(tt_low>=t_start*t[1])]
        tt_low = tt_low[np.where(tt_low>=t_start*t[1])]
        n_low_avg = (1/(tt_low[-1]-tt_low[0])) * np.trapz(n_low, tt_low)

        n_high = n_high[np.where(tt_high>=t_start*t[1])]
        tt_high = tt_high[np.where(tt_high>=t_start*t[1])]
        n_high_avg = (1/(tt_high[-1]-tt_high[0])) * np.trapz(n_high, tt_high)

        if n>n_high_avg:
            raise ValueError('I_guess[1] not large enough')
        elif n<n_low_avg:
            raise ValueError('I_guess[0] not small enough')

        relative_difference = np.abs((n-n_low_avg)/n)
        if any(relative_difference<=tol):
            index = np.argmin(relative_difference-tol)
            found_list[index] = True
            I_list[index] = I_guess[0]

        relative_difference = np.abs((n-n_high_avg)/n)
        if any(relative_difference<=tol):
            index = np.argmin(relative_difference - tol)
            found_list[index] = True
            I_list[index] = I_guess[1]

        I_search_array = np.copy(I_guess)
        I_trys = np.copy(I_guess)

        while not all(found_list):
            log_of_I = \
                    (np.log10(I_search_array[0])+np.log10(I_search_array[1]))/2.0

            I_curr = np.array([10**log_of_I])
            I_trys = np.concatenate((I_trys, I_curr))

            self.I_func = lambda t: I_curr

            tt, n_curr = self.run(option=option)
            n_curr = n_curr[np.where(tt>=t_start*t[1])]
            tt = tt[np.where(tt>=t_start*t[1])]

            n_curr_avg = (1/(tt[-1]-tt[0])) * np.trapz(n_curr, tt)

            relative_difference = np.abs((n-n_curr_avg)/n)
            if any(relative_difference <= tol):
                index = np.argmin(relative_difference-tol)
                found_list[index] = True
                I_list[index] = I_curr

            if any(n>n_curr_avg):
                I_search_array[0] = I_curr
            else:
                I_search_array[1] = I_curr

            print n_curr_avg, n

        return I_list, I_trys

##############################################################
### NOT WORKING BELOW DO NOT USE
##############################################################

class LaserRateEquations(RateEquations):
    def __init__(self):
        pass

    def get_gain(self, Nq):
        N = np.squeeze(self.LED.active_mat.N)
        P = np.squeeze(self.LED.active_mat.P)

        gain = np.squeeze(self.LED.active_mat.gain)
        gain = np.squeeze(gain[4600,:])

        import matplotlib.pyplot as plt
        f=plt.figure()
        ax = f.add_subplot(111)
        ax.semilogx(N, gain/100)
        plt.show()

        gain_int = np.interp(Nq, N, gain)
        return gain_int

    def compute_RHS_S(self, t, y):
        #print y, y[0], y[1]
        if t <= self.t_ramp:
            RHS1 = self.etai/(q*self.V) * self.I_func_ramp(t) - \
                    self.get_Rspon(y[0]) - self.get_Rnr(y[0]) - \
                    self.LED.vg * self.get_gain(y[0]) * y[1]
        else:
            RHS1 = self.etai/(q*self.V) * self.I_func(t) - \
                    self.get_Rspon(y[0]) - self.get_Rnr(y[0]) - \
                    self.LED.vg * self.get_gain(y[0]) * y[1]

        RHS2 = self.LED.Gamma * self.LED.vg * self.get_gain(y[0]) * y[1] + \
                self.LED.Gamma * self.LED.beta * self.get_Rspon(y[0]) - \
                y[1] / self.LED.tau_p

        return np.array([RHS1, RHS2])

    def calc_peak_to_peak_power_S(self, t, Nt, St, t_start):
        deltaP_N = (1-self.LED.beta) * \
                self.calc_peak_to_peak_power(self, np.squeeze(t), np.squeeze(Nt), t_start)

        St = St[np.where(t>=t_start)]
        t = t[np.where(t>=t_start)]

        max_St = np.amax(St)
        min_St = np.amin(St)

        max_hvStaup = self.LED.active_mat.Eg * max_St / self.LED.tau_p
        min_hvStaup = self.LED.active_mat.Eg * min_St / self.LED.tau_p

        deltaP_S = max_hvStaup - min_hvStaup

        return deltaP_N + deltaP_S


if __name__ == '__main__':
    import LED_new
    import active_material

    n = 3.5
    Egw = 0.8*q
    Ep = 25.7 * q
    me = 0.067 * m0
    mh = 0.47 * m0
    Nd = 0.0
    Na = 0.0
    Lz = 10.0e-9
    M = (m0/6)*Ep
    mw = 0.067*m0
    mw_lh = 0.087*m0
    mw_hh = 0.5*m0

    omega = np.linspace(0.1, 2.5, num=2000)*q/hbar
    DF_max = 1.5*q
    DF_dis = 1.5*q/500
    T=300.0
    

    QW = active_material.QuantumWell(omega, DF_max, DF_dis, T, n, mw, mw_lh, mw_hh, Ep, M, Egw,
                     Na, Nd, Lz)

    QW.build()

    Fx = 5*np.ones(omega.size)
    antenna = LED_new.Antenna(omega, Fx)

    nanoLED = LED_new.nanoLED_QW(omega, QW.DF, T, QW, antenna)
    nanoLED.build()

    
    ns_master = 1e6*5e18
    t_dummy = [0.0, 1e-6]
    I_func_dummy = lambda t:  I0[0]

    reqns = RateEquations(nanoLED, t_dummy, I_func_dummy, V=1e-18, N0 =[1e17*1e6], max_step=1e-10 )


    I_ant, _ = reqns.find_desired_Idc(t_dummy, 1.0e-11, ns_master, np.array([5e-10, 5e0]),'nanoLED', tol=1e-4)
    I_func = lambda t: 0
    t_ramp = 2e-7
    I_func_ramp = lambda t: I_ant

    max_step = 0.5e-11

    time = [0.0, 8e-7]
    reqns.update(time, I_func, I_func_ramp = I_func_ramp, t_ramp = t_ramp,
                 V=1e-18, N0=[1e23], max_step = max_step)
    tt, Nt = reqns.run(option='nanoLED')


    import matplotlib.pyplot as plt
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(tt, Nt/1e6)

    plt.show()
