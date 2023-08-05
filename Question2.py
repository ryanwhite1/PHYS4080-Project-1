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

def dGdx(g, x, m, spin, cross, log):
    '''
    '''
    if log:
        ov = cms2gev(10**cross)
    else:
        ov = cms2gev(cross)
    ans = (m / x**2) * ov * ((Yeq(x, spin) * mu)**2 - g**2)
    return ans

def dimenless_abund(xarr, m, spin, cross, log):
    '''
    '''
    g0 = mu * Yeq(xarr[0], spin)
    ans = integ.odeint(dGdx, g0, xarr, args=(m, spin, cross, log)) / mu
    return ans

def cosmo_abund(cross, xarr, m, spin, shift, log):
    '''
    '''
    mult = 1 / (3.63 * 10**-9)
    return dimenless_abund(xarr, m, spin, cross, log)[-1] * m * mult - shift

def root_find(xarr, mass):
    '''
    '''
    root1 = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0.12 + 0.003, True), maxfev=10000)
    root2 = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0.12 - 0.003, True), maxfev=10000)
    frac = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0., True), maxfev=10000)
    return root1, root2, frac


### Q3a ###

# x = np.logspace(0, 3, 100)

# fig, ax = plt.subplots(figsize=(4.5, 5))

# masses = [100, 100, 100, 100, 100, 10]
# ovs = [-26., -27., -28.1, -29., -30., -30.]

# for i, mass in enumerate(masses):
#     y = dimenless_abund(x, mass, 1/2, ovs[i], True)
#     ax.plot(x, y, label=f'$m_\chi$={mass}, $\log_{{10}}(\sigma v)={ovs[i]}$')
# ax.set_xscale('log'); ax.set_yscale('log')
# ax.set_xlabel("$x = m / T$"); ax.set_ylabel("Abundance $Y$")
# ax.legend(loc='upper right')
# ax.set_ylim(ymax=0.1)

# fig.savefig("Q3a.png", dpi=400, bbox_inches='tight')
# fig.savefig("Q3a.pdf", dpi=400, bbox_inches='tight')


# ### Now Q3b ###
# x = np.logspace(1, 3, 100)
# masses = np.logspace(-3, 4, 600)
# ovs = np.linspace(-30, -26, 5)

# ys = np.zeros((len(masses), len(ovs)))

# for i, mass in enumerate(masses):
#     for j, ov in enumerate(ovs):
#         ys[i, j] = cosmo_abund(ov, x, mass, 1/2, 0, True)

# fig, ax = plt.subplots(figsize=(8, 4))
# for i in range(len(ovs)):
#     ax.plot(masses, ys[:, i], label=f'$\log_{{10}}(\sigma v) = {ovs[i]}$')
# # ax.set_xscale('log')
# ax.set_yscale('log'); ax.set_xscale('log')
# ax.set_ylim(ymin=0.1)
# ax.legend(loc='upper left')
# ax.set_xlabel("Mass $m_\chi$ (GeV)")
# ax.set_ylabel("Present Day Abundance $\Omega h^2$")
# fig.savefig("Q3b.png", dpi=400, bbox_inches='tight')
# fig.savefig("Q3b.pdf", dpi=400, bbox_inches='tight')



### Q3c ###

# x = np.logspace(1.1, 3, 400)
# masses = np.logspace(0, 4, 100)

# ovs = np.empty((len(masses), 3))
# for i, mass in enumerate(masses):
#     ovs[i, :] = root_find(x, mass)
    
# fig, ax = plt.subplots(figsize=(4.5, 4))

# ax.fill_between(masses, 10**ovs[:, 0], 10**ovs[:, 1], alpha=0.8, color='tab:blue', label='$3\sigma$ Region')
# ax.fill_between(masses, 10**ovs[:, 1], 10**ovs[:, 2], alpha=0.3, color='tab:red', label='Partial Solution')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_ylabel("Cross Section $<\sigma v>$ (cm$^3$/s)", usetex=True)
# ax.set_xlabel("Mass $m_\chi$ (GeV)")
# ax.legend()

# fig.savefig('Q3c.png', dpi=400, bbox_inches='tight')
# fig.savefig('Q3c.pdf', dpi=400, bbox_inches='tight')

# ax.set_ylim([0.9 * min(10**ovs[:, 1]), 1.1 * max(10**ovs[:, 0])])
# ax.legend(loc='lower right')
# fig.savefig('Q3c-zoom.png', dpi=400, bbox_inches='tight')
# fig.savefig('Q3c-zoom.pdf', dpi=400, bbox_inches='tight')


### Q4a ###







