# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:07:52 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.integrate as integ

h_eff = 86.25
gstar = 86.25**(1/2)
Mpl = 1.2 * 1e19 # GeV
mu = np.sqrt(np.pi / 45) * Mpl * gstar**(1/2)

def cms2gev(ov):
    return ov / (1.17 * 1e-17)

def Yeq(x, spin):
    '''
    '''
    ans = (45 / (4 * np.pi**4)) * (x**2 / h_eff) * (2 * spin + 1) * sp.kn(2, x)
    return ans

def dGdx(g, x, m, spin, cross):
    '''
    '''
    ans = (m / x**2) * cross * ((Yeq(x, spin) * mu)**2 - g**2)
    return ans

def cos_dens(xarr, m, spin, cross):
    '''
    '''
    g0 = mu * Yeq(xarr[0], spin)
    ans = integ.odeint(dGdx, g0, xarr, args=(m, spin, cross)) / mu
    return ans

x = np.logspace(0, 3, 100)

y = cos_dens(x, 100, 1/2, cms2gev(1e-26))
y2 = cos_dens(x, 100, 1/2, cms2gev(1e-27))
yeq = Yeq(x, 1/2)


fig, ax = plt.subplots()

ax.plot(x, y)
ax.plot(x, y2)
ax.set_xscale('log')
ax.set_yscale('log')

