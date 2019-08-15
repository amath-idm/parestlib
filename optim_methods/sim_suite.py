'''
Like problem_suite.py, but more complicated simulations.

See the tests folder for example usages.

Version: 2019aug14
'''

import pylab as pl
import scipy.signal as si
import sciris as sc

__all__ = ['blowfly_sim', 'make_blowflies', 'plot_blowflies']


def blowfly_sim(pars, initialpop=None, npts=None):
    '''
    Nicholson's blowfly population model. Inspired by example in:
        Hsu K, Ramos F. 
        Bayesian Learning of Conditional Kernel Mean Embeddings for Automatic Likelihood-Free Inference. 
        March 2019. http://arxiv.org/abs/1903.00863.
    
    Code based on:
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


def blowfly_statistics(y):
    mean = pl.mean(y)
    skew = pl.median(y) - mean
    yzeromean = y-mean
    ysmooth = sc.smooth(yzeromean, 1)
    autocorr = pl.correlate(ysmooth, ysmooth, mode='same')
    autocorr /= autocorr.max()
    freqs,spectrum = si.periodogram(autocorr)
    stats = {'mean':mean, 'skew':skew, 'autocorr':autocorr, 'freqs':freqs, 'spectrum':spectrum, 'ysmooth':ysmooth}
    return stats
    


def make_blowflies(noise=0.0, optimum='min', verbose=True):
    
    def blowfly_err():
        pass
    
    func = lambda pars: blowfly_err(pars) # Create the actual function
    
    if verbose:
        print("Created blowfly function with noise=%s" % (noise))
        print('Suggested starting point: [0.5,0.5]')
        print('Optimal solution: %s≈1.037 near [4,1]' % optimum)
    return func


def plot_blowflies(pars=(pl.exp(3.8), 0.3), initialpop=None, npts=400):
    x = pl.arange(npts)
    y = blowfly_sim(pars=pars, initialpop=initialpop, npts=npts)
    stats = blowfly_statistics(y)
    
    pl.figure()
    
    pl.subplot(3,1,1)
    pl.plot(x, stats['ysmooth'], marker='o')
    pl.xlabel('Days')
    pl.ylabel('Population size')
    pl.title('Blowfly simulation with r=%0.2f, σ=%0.2f' % (pars[0], pars[1]))
    
    pl.subplot(3,1,2)
    pl.plot(stats['autocorr'])
    
    pl.subplot(3,1,3)
    pl.plot(stats['spectrum'])    
    
    output = {'x':x, 'y':y}
    output.update(stats)
    return output
    