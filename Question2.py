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

# set matplotlib plotting params to use latex
plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

# define constants needed in models
h_eff = 86.25                                   # effective entropic dof
gstar = 86.25**(1/2)                            # effective energetic dof
Mpl = 1.2 * 1e19                                # planck mass, GeV
mu = np.sqrt(np.pi / 45) * Mpl * gstar**(1/2)   # scaling constant

def cms2gev(ov):
    '''Converts from units of cm^3/s to GeV'''
    return ov / (1.17 * 1e-17)

def Yeq(x, spin):
    ''' Returns the equilibrium abundance of a particle given by Maxwell-Boltzmann approximation
    Inputs
    ------
    x : float
        Effective temperature of the universe (x = particle mass / temperature)
    spin : float
        Particle spin
    '''
    return (45 / (4 * np.pi**4)) * (x**2 / h_eff) * (2 * spin + 1) * sp.kn(2, x)

def dGdx(g, x, m, spin, cross, log):
    ''' Scaled Boltzmann equation, giving the particle abundance at `time' x, where dG/dx = mu * dY/dx
    Inputs
    ------
    g : float
        Scaled initial condition of the ODE; g = mu * y
    x : float
        Effective temperature of the universe (x = particle mass / temperature)
    m : float
        Particle mass (in GeV)
    spin : float
        Particle spin
    cross : float
        Interaction cross section (in cm^3/s) of the particle. If `log` == True, then this should be log10(ov)
    '''
    ov = cms2gev(10**cross) if log else cms2gev(cross)
    return (m / x**2) * ov * ((Yeq(x, spin) * mu)**2 - g**2)

def dimenless_abund(xarr, m, spin, cross, log):
    ''' Solves the ODE dY/dx to give an approximate value for Y (particle abundance) at each x in xarr. 
    Inputs
    ------
    xarr : np.array
        Array of effective temperatures of the universe (x = particle mass / temperature)
    m : float
        Particle mass (in GeV)
    spin : float
        Particle spin
    cross : float
        Interaction cross section (in cm^3/s) of the particle. If `log` == True, then this should be log10(ov)
    '''
    g0 = mu * Yeq(xarr[0], spin) # scaled initial condition
    ans = integ.odeint(dGdx, g0, xarr, args=(m, spin, cross, log)) / mu # calculate the ODE and rescale it back from G to Y
    return ans

def cosmo_abund(cross, xarr, m, spin, shift, log):
    ''' Effectively the same as the function dimenless_abund, except scales the last value to give the approx. present day value. 
    Inputs
    ------
    cross : float
        Interaction cross section (in cm^3/s) of the particle. If `log` == True, then this should be log10(ov)
    xarr : np.array
        Array of effective temperatures of the universe (x = particle mass / temperature)
    m : float
        Particle mass (in GeV)
    spin : float
        Particle spin
    shift : float
        A parameter useful when root finding. Is the `desired` present day abundance. If not root finding, then set to 0. 
    '''
    mult = 1 / (3.63 * 10**-9)
    return dimenless_abund(xarr, m, spin, cross, log)[-1] * m * mult - shift

def root_find(xarr, mass):
    ''' Finds the regions in parameter space where a spin 1/2 particle could yield the observed Planck dark matter density.
    Inputs
    ------
    xarr : np.array
        Array of effective temperatures of the universe (x = particle mass / temperature). (Used to find present day value)
    mass : float
        Particle mass (in GeV)
    Returns
    -------
    root1 : float
        The +3 sigma upper bound on the cross-section parameter to explain the Planck CDM density with a particle of mass `mass`
    root2 : float
        The -3 sigma upper bound on the cross-section parameter to explain the Planck CDM density with a particle of mass `mass`
    frac : float
        The (very) rough upper bound on cross section to explain *any* of the observed dark matter with our spin 1/2 particle.
    '''
    root1 = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0.12 + 0.003, True), maxfev=10000) # start with initial guess of ov = 1e-25
    root2 = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0.12 - 0.003, True), maxfev=10000) # have more iters to help the awful ODE converge
    frac = opt.fsolve(cosmo_abund, -25., args=(xarr, mass, 1/2, 0., True), maxfev=10000)
    return root1, root2, frac


### Q3a ###

# x = np.logspace(0, 3, 100)  # initialise temps

# fig, ax = plt.subplots(figsize=(4.5, 5))    # make figure

# # define parameters we want to look at
# masses = [100, 100, 100, 100, 100, 10]
# ovs = [-26., -27., -28.1, -29., -30., -30.]

# for i, mass in enumerate(masses):
#     y = dimenless_abund(x, mass, 1/2, ovs[i], True)     # solve for the abundance across our temps
#     ax.plot(x, y, label=f'$m_\chi$={mass}, $\log_{{10}}(\sigma v)={ovs[i]}$')   # and plot it on the figure
    
# ax.set_xscale('log'); ax.set_yscale('log')
# ax.set_xlabel("$x = m / T$"); ax.set_ylabel("Abundance $Y$")
# ax.legend(loc='upper right')
# ax.set_ylim(ymax=0.1)   # set a bit higher so that the legend fits nicely

# fig.savefig("Q3a.png", dpi=400, bbox_inches='tight')    # save figures as png (to look at) and pdf (for the report)
# fig.savefig("Q3a.pdf", dpi=400, bbox_inches='tight')


# ### Now Q3b ###
# x = np.logspace(1, 3, 100)
# masses = np.logspace(-3, 4, 600) # big range of masses to look at
# ovs = np.linspace(-30, -26, 5) # look at a few cross sections - this is a linspace, but these values will be used as powers

# ys = np.zeros((len(masses), len(ovs))) # initialise array of data

# for i, mass in enumerate(masses):
#     for j, ov in enumerate(ovs):
#         ys[i, j] = cosmo_abund(ov, x, mass, 1/2, 0, True) # calculate present day val for this mass and cross section, and store it

# # now to plot these present day vals
# fig, ax = plt.subplots(figsize=(8, 4)) # wide figure!
# for i in range(len(ovs)):
#     ax.plot(masses, ys[:, i], label=f'$\log_{{10}}(\sigma v) = {ovs[i]}$')
    
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

# ovs = np.empty((len(masses), 3)) # initialise empty array for our data
# for i, mass in enumerate(masses):
#     ovs[i, :] = root_find(x, mass) # find the 3 cross section params for this mass and store them
    
# fig, ax = plt.subplots(figsize=(4.5, 4))
# # want to plot filled in regions, and since our ovs are powers, we need to exponentiate them to get meaningful, physical values
# ax.fill_between(masses, 10**ovs[:, 0], 10**ovs[:, 1], alpha=0.8, color='tab:blue', label='$3\sigma$ Region')
# ax.fill_between(masses, 10**ovs[:, 1], 10**ovs[:, 2], alpha=0.3, color='tab:red', label='Partial Solution')
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_ylabel("Cross Section $<\sigma v>$ (cm$^3$/s)", usetex=True)
# ax.set_xlabel("Mass $m_\chi$ (GeV)")
# ax.legend()

# fig.savefig('Q3c.png', dpi=400, bbox_inches='tight')
# fig.savefig('Q3c.pdf', dpi=400, bbox_inches='tight')

# # now save a zoomed in plot so that we can get a better look at the +/- 3 sigma region
# ax.set_ylim([0.9 * min(10**ovs[:, 1]), 1.1 * max(10**ovs[:, 0])])
# ax.legend(loc='lower right')
# fig.savefig('Q3c-zoom.png', dpi=400, bbox_inches='tight')
# fig.savefig('Q3c-zoom.pdf', dpi=400, bbox_inches='tight')


### Q4a ###







