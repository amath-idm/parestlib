'''
Like problem_suite.py, but more complicated simulations.

See the tests folder for example usages.

Version: 2019aug14
'''

import pylab as pl
import scipy.signal as si
import sciris as sc
import optim_methods as om

__all__ = ['blowfly_sim', 'make_blowflies', 'plot_blowflies']

# Set default parameters
default_blowfly_pars = (pl.exp(3.8), 0.7)
default_initpop = 1
default_npts = 1000


def blowfly_sim(pars, initpop=None, npts=None):
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
    if pars    is None: pars    = default_blowfly_pars # Growth rate and noise term
    if initpop is None: initpop = default_initpop # Population size
    if npts    is None: npts    = default_npts # Number of time points
    
    # Set parameters
    r = pars[0]
    σ = pars[1]
    y = pl.zeros(npts)
    y[0] = initpop
    
    # Run simulation
    for t in range(npts-1):
        Pn = y[t]
        ε = σ*pl.randn()
        Pn1 = r*Pn*pl.exp(-Pn+ε)
        y[t+1] = Pn1
    
    return y


def blowfly_statistics(y):
    mean = pl.mean(y)
    skew = pl.median(y) - mean
    yzeromean = y-mean
    autocorr = pl.correlate(yzeromean, yzeromean, mode='same')
    autocorr /= autocorr.max()
    freqs,spectrum = si.periodogram(autocorr)
    cumdist = pl.sort(y)
    stats = sc.objdict({'cumdist':cumdist, 'mean':mean, 'skew':skew, 'autocorr':autocorr, 'freqs':freqs, 'spectrum':spectrum})
    return stats
    

def make_blowflies(noise=0.0, optimum='min', verbose=True):
    
    default_y = blowfly_sim(pars=default_blowfly_pars, initpop=None, npts=None)
    default_stats = blowfly_statistics(default_y)
    
    def blowfly_err(pars):
        y = blowfly_sim(pars=pars, initpop=None, npts=None)
        stats = blowfly_statistics(y)
        mismatch = stats['cumdist'] - default_stats['cumdist']
        err = pl.sqrt(pl.mean(mismatch**2)) # Calculate RMSE between predicted and actual CDF
        err = om.problem_suite.addnoise(err, noise)
        if optimum == 'max':
            err = -err
        return err
    
    func = lambda pars: blowfly_err(pars) # Create the actual function
    
    if verbose:
        print("Created blowfly function with noise=%s" % (noise))
        print('Suggested starting point: [20,0.5]')
        print('Suggested limits: r ~ [0,80] and σ ~ [0,2]')
        print('Optimal solution: %s≈0.2 near %s' % (optimum, str(default_blowfly_pars)))
    return func


def plot_blowflies(pars=default_blowfly_pars, initpop=None, npts=default_npts, fig=None):
    x = pl.arange(npts)
    y = blowfly_sim(pars=pars, initpop=initpop, npts=npts)
    stats = blowfly_statistics(y)
    
    # Allow points to be added to an existing figure
    if fig is None:
        fig = pl.figure()
    
    if len(fig.axes)<2:
        ax1 = pl.subplot(2,1,1)
        ax2 = pl.subplot(2,1,2)
    else:
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        
    ax1.plot(x, y, marker='o', lw=2)
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Population size')
    ax1.set_title('Blowfly simulation with r=%0.2f, σ=%0.2f' % (pars[0], pars[1]))
    
    ax2.plot(x, stats['cumdist'], marker='o', lw=2)
    ax2.set_xlabel('Order')
    ax2.set_ylabel('Population size')
    ax2.set_title('Population distribution')
    
    output = sc.objdict({'x':x, 'y':y})
    output.update(stats)
    return output
    