# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:07:52 2023

@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import scipy.integrate as integ
import scipy.optimize as opt

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

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

def dimenless_abund(xarr, m, spin, cross):
    '''
    '''
    g0 = mu * Yeq(xarr[0], spin)
    ans = integ.odeint(dGdx, g0, xarr, args=(m, spin, cross)) / mu
    return ans

def root_find(xarr, mass, ovrange):
    mult = 1 / (3.63 * 10**-9)
    func = lambda x, b: dimenless_abund(xarr, mass, 1/2, x)[-1] * mult * mass - 0.1200 + b
    zero = lambda x: dimenless_abund(xarr, mass, 1/2, x)[-1] * mult * mass
    
    print(func(ovrange[0], 0.001), func(ovrange[1], 0.001))
    root1 = opt.brentq(func, ovrange[0], ovrange[1], args=(0.001))
    root2 = opt.brentq(func, ovrange[0], ovrange[1], args=(-0.001))
    frac = opt.brentq(zero, ovrange[0], ovrange[1])
    return [root1, root2, frac]

x = np.logspace(0, 3, 100)


### Q3a ###

# fig, ax = plt.subplots()

# masses = [100, 50, 100, 100, 20]
# ovs = [1e-26, 1e-26, 1e-27, 1e-30, 1e-30]

# for i, mass in enumerate(masses):
#     y = dimenless_abund(x, mass, 1/2, cms2gev(ovs[i]))
#     ax.plot(x, y, label=f'$m_\chi$={mass}GeV, $\sigma v$={ovs[i]}')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel("$x = m / T$")
# ax.set_ylabel("Abundance $Y$")
# ax.legend()

# fig.savefig("Q3a.png", dpi=400, bbox_inches='tight')
# fig.savefig("Q3a.pdf", dpi=400, bbox_inches='tight')


# ### Now Q3b ###
# masses = np.arange(1, 200, 1)
# # masses = np.logspace(-5, 2, 20)
# ovs = np.logspace(-30, -26, 5)
# ys = np.zeros((len(masses), len(ovs)))
# mult = 1 / (3.63 * 10**-9)

# for i, mass in enumerate(masses):
#     for j, ov in enumerate(ovs):
#         ys[i, j] = dimenless_abund(x, mass, 1/2, cms2gev(ov))[-1] * mult * mass

# fig, ax = plt.subplots()
# for i in range(len(ovs)):
#     ax.plot(masses, ys[:, i], label=f'$\sigma v = ${ovs[i]}')
# # ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_ylim(ymin=0.1)
# ax.legend()
# ax.set_xlabel("Mass $m_\chi$ (GeV)")
# ax.set_ylabel("Present Day Abundance $\Omega h^2$")
# fig.savefig("Q3b.png", dpi=400, bbox_inches='tight')
# fig.savefig("Q3b.pdf", dpi=400, bbox_inches='tight')



### Q3c ###

x = [23.658, 1000]

fig, ax = plt.subplots()

masses = np.linspace(20, 100, 10)
orange = [1e-45, 1e-20]

ovs = np.empty((len(masses), 3))
for i, mass in enumerate(masses):
    ovs[i, :] = root_find(x, mass, orange)







