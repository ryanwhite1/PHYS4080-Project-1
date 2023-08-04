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
    # ans = integ.solve_ivp(dGdx, [xarr[0], xarr[-1]], [g0], method='BDF', t_eval=xarr, args=(m, spin, cross, log)).y[-1] / mu
    return ans

def cosmo_abund(cross, xarr, m, spin, shift, log):
    '''
    '''
    mult = 1 / (3.63 * 10**-9)
    return dimenless_abund(xarr, mass, spin, cross, log)[-1] * m * mult - shift

def root_find(xarr, mass, ovrange):
    '''
    '''
    # o1, o2 = cms2gev(ovrange[0]), cms2gev(ovrange[1])
    # o1, o2 = ovrange
    o1, o2 = np.log10(ovrange)
    # print(o1, o2)
    print(cosmo_abund(o1, xarr, mass, 1/2, 0.12 + 0.003, True), cosmo_abund(o2, xarr, mass, 1/2, 0.12 + 0.003, True))
    # print(opt.fsolve(cosmo_abund, cms2gev(1e-25), args=(xarr, mass, 1/2, 0), maxfev=10000) * (1.17 * 1e-17))
    
    root1 = opt.brentq(cosmo_abund, o1, o2, args=(xarr, mass, 1/2, 0.12 + 0.003, True), maxiter=10000)
    root2 = opt.brentq(cosmo_abund, o1, o2, args=(xarr, mass, 1/2, 0.12 - 0.003, True), maxiter=10000)
    print(root1, root2)
    # frac = opt.brentq(cosmo_abund, o1, o2, args=(xarr, mass, 1/2, 0, True))
    frac = 0
    # root1 = opt.fsolve(cosmo_abund, cms2gev(1e-25), args=(xarr, mass, 1/2, -0.12 - 0.003), maxfev=10000) * (1.17 * 1e-17)
    # root2 = opt.fsolve(cosmo_abund, cms2gev(1e-25), args=(xarr, mass, 1/2, -0.12 + 0.003), maxfev=10000) * (1.17 * 1e-17)
    # frac = opt.fsolve(cosmo_abund, cms2gev(1e-25), args=(xarr, mass, 1/2, 0), maxfev=10000) * (1.17 * 1e-17)
    return root1, root2, frac

# x = np.logspace(0, 3, 100)


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
# x = np.logspace(1, 3, 100)
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

# x = [23.658, 1000]

# fig, ax = plt.subplots()

x = np.logspace(1, 3, 200)
# x = [10, 1000]
# masses = np.linspace(50, 100, 100)
masses = np.logspace(1.5, 3, 100)
orange = [1e-30, 1e-10]

ovs = np.empty((len(masses), 3))
for i, mass in enumerate(masses):
    # try:
    #     ovs[i, :] = root_find(x, mass, orange)
    # except:
    #     pass
    ovs[i, :] = root_find(x, mass, orange)
    
fig, ax = plt.subplots()

ax.fill_between(masses, ovs[:, 0], ovs[:, 1], alpha=0.8, color='tab:blue')
ax.fill_between(masses, ovs[:, 0], ovs[:, 2], alpha=0.3, color='tab:red')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel("Cross Section $<\sigma v>$ (cm$^3$/s)")
ax.set_xlabel("Mass $m_\chi$ (GeV)")






