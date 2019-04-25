#!/usr/bin/env python
"""
Provides a few useful functions
"""

from math import pi
import numpy as np
from fdint import fdk

from physical_constants import *

__author__ = "Sean Hooten"
__license__ = "BSD-2-Clause"
__version__ = "0.1"
__maintainer__ = "Sean Hooten"
__status__ = "development"

def fermi_dirac_integral(k, phi):
    if k==0.5:
        gamma = np.sqrt(pi)/2
    elif k==0:
        gamma = 1
    elif k==-0.5:
        gamma = sqrt(pi)
    else:
        raise ValueError('k is not the correct value')

    fd = (1/gamma)*fdk(k, phi)
    return fd

def sinusoid(wt, A, phi, offset):
    return A*np.cos(wt+phi)+offset

def lineshape_broadening(omega, spectrum, t_in):
   # Lorentzian
   broadened = np.zeros(omega.size)
   const = hbar/t_in
   for i in range(omega.size):
       L = (1/pi) * (const) / (const**2 + hbar**2 * (omega - omega[i])**2)
       broadened[i] = np.trapz(spectrum*L, x=hbar*omega)

   return broadened

#def lineshape_broadening(self, gain):
        #pass
        # using Chinn lineshape function
        # assumes omega has constant spacing

        #omega = self.omega

        #N = omega.size
        #delF = (omega[1] - omega[0])/(2*pi)

        #time_res = 1 / (2*delF*(N-1)) # due to real ifft
        #time_max = 1 / (delF)
        #times = np.linspace(0, time_max, num=2*(N-1))

        #ifft_gain = np.fft.irfft(gain)

        #times_ps = times*1e12

        #log10t = 2 + 1.5*np.log10(times_ps) - \
        #        0.5*np.sqrt((2+np.log10(times_ps))**2 + 0.36)

        #l_t = 10**log10t

        #multiplier = np.exp(-1*l_t)

        #gain_broadened = np.fft.rfft(ifft_gain * multiplier, n = 2*N-1)


        #omega = self.omega


        ## Lorentzian
        #omega = self.omega
        #gain_broadened = np.zeros(omega.size)

        #tm = 0.1e-12
        #const = hbar/tm
        #for i in range(omega.size):

        #    L = (1/pi) * (const) / (const**2 + hbar**2 * (omega - omega[i])**2)

        #    gain_broadened[i] = np.trapz(gain*L, x=hbar*omega)

        #return gain_broadened
