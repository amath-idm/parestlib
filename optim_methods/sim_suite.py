'''
Like problem_suite.py, but more complicated simulations.

See the tests folder for example usages.

Version: 2019aug14
'''

import pylab as pl

__all__ = ['blowfly_sim', 'plot_blowflies']


def blowfly_sim(pars, initialpop=1000, npts=400):
    '''
    Nicholson's blowfly population model. Based on:
        https://www.nature.com/articles/nature09319
        https://wiki.its.sfu.ca/research/datagroup/images/4/44/Wood_nature.pdf
        https://ionides.github.io/531w16/final_project/Project18/Finalprojectv2.html
        https://rdrr.io/github/kingaa/pomp/man/blowflies.html
        https://github.com/kingaa/pomp/blob/master/R/blowflies.R
    '''
    # Handle input arguments -- default values from Wood's paper
    if pars       is None: pars       = [pl.exp(3.8), 0.3] # Growth rate and noise term
    if initialpop is None: initialpop = 1 # Population size
    if npts       is None: npts       = 400 # Number of time points
    
    # Set parameters
    r = pars[0]
    σ = pars[1]
    y = [initialpop]
    
    # Run simulation
    for t in range(npts-1):
        Pn = y[-1]
        ε = σ*pl.randn()
        Pn1 = r*Pn*pl.exp(-Pn+ε)
        y.append(Pn1)
    
    return y


def plot_blowflies(pars=(pl.exp(3.8), 0.3), initialpop=None, npts=400):
    x = pl.arange(npts)
    y = blowfly_sim(pars=pars, initialpop=initialpop, npts=npts)
    pl.figure()
    pl.plot(x, y, marker='o')
    pl.xlabel('Days')
    pl.ylabel('Population size')
    pl.title('Blowfly simulation with r=%0.2f, σ=%0.2f' % (pars[0], pars[1]))
    return y
    