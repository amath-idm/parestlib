'''
Like problem_suite.py, but more complicated simulations.

See the tests folder for example usages.

Version: 2019sep20
'''

import pylab as pl
import scipy.signal as si
import scipy.stats as st
import sciris as sc
import parestlib as om

__all__ = ['blowfly_sim', 'make_blowflies', 'plot_blowflies', 'SIR_sim', 'make_SIR', 'plot_SIR']



###############################################################################
#%% Blowfly simulation
###############################################################################

def blowfly_sim(pars=None, initpop=None, npts=None):
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
    
    Parameters
    ----------
    pars : tuple
        Parameters of the simulation: pars[0]=r is the reproduction rate, and pars[1]=σ
        is the variability.
    
    initpop : int
        Initial population size (number of flies)
    
    npts : int
        Number of time points to run the simulation for
    
    Returns
    -------
    y : array
        Time series of population size
    
    Example
    -------
    import parestlib as om
    y = om.blowfly_sim(pars=(pl.exp(3.8), 0.7), initpop=1, npts=1000)
    
    '''
    # Set default parameters
    default_pars = [pl.exp(3.8), 0.7]
    default_initpop = 1
    default_npts = 1000

    # Handle input arguments -- default values from Wood's paper
    if pars    is None: pars    = default_pars # Growth rate and noise term
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
    '''
    Calculate statistics from the blowfly population time series, used for evaluating the likelihood.
    '''
    mean = pl.mean(y)
    skew = pl.median(y) - mean
    yzeromean = y-mean
    autocorr = pl.correlate(yzeromean, yzeromean, mode='same')
    autocorr /= autocorr.max()
    freqs,spectrum = si.periodogram(autocorr)
    cumdist = pl.sort(y)
    stats = sc.objdict({'cumdist':cumdist, 'mean':mean, 'skew':skew, 'autocorr':autocorr, 'freqs':freqs, 'spectrum':spectrum})
    return stats
    

def make_blowflies(noise=0.0, optimum='min', verbose=True, **kwargs):
    '''
    Create the actual blowflies object for use in optimization algorithms -- returns
    a function that when called returns a likelhood/goodness-of-fit.
    
    Example
    -------
    import scipy
    initpars = [2.0, 0.7]
    blowfly_func = om.make_blowflies()
    outpars = scipy.minimize(blowfly_func, initpars)
    '''
    default_y = blowfly_sim(**kwargs)
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
#        print('Optimal solution: %s≈0.2 near %s' % (optimum, str(default_blowfly_pars)))
    return func


def plot_blowflies(pars=None, initpop=None, npts=None, fig=None):
    '''
    Plot the time series and summary statistics for a given blowfly simulation.
    
    Example
    -------
    om.plot_blowflies()
    '''
    y = blowfly_sim(pars=pars, initpop=initpop, npts=npts)
    x = pl.arange(len(y))
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
#    ax1.set_title('Blowfly simulation with r=%0.2f, σ=%0.2f' % (pars[0], pars[1])) # TODO: fix broken for pars=None
    
    ax2.plot(x, stats['cumdist'], marker='o', lw=2)
    ax2.set_xlabel('Order')
    ax2.set_ylabel('Population size')
    ax2.set_title('Population distribution')
    
    output = sc.objdict({'x':x, 'y':y})
    output.update(stats)
    return output
    


###############################################################################
#%% Stochastic SIR simulation
###############################################################################

def SIR_sim(pars, popsize=None, npts=None):
    '''
    Stochastic SIR model.
    
    Version: 2019sep20
    '''
    # Set default parameters
    default_pars = [0.10, 20, 0.1] # Force of infection, average duration of infection (days), initial fraction infected
    default_popsize = 100
    default_npts = 365
    
    # Handle input arguments -- default values from Wood's paper
    if pars    is None: pars    = default_pars # Growth rate and noise term
    if popsize is None: popsize = default_popsize # Population size
    if npts    is None: npts    = default_npts # Number of time points
    
    # Set parameters
    foi = pars[0]
    duration = pars[1]
    init_infected = st.binom.rvs(n=popsize, p=pars[2], size=1)[0]
    
    # Initial conditions
    y = sc.dataframe(cols=['s','i','r'], nrows=npts)
    y['s', 0] = popsize-init_infected
    y['i', 0] = init_infected
    
    # Run simulation
    for t in range(npts-1):
        p_infection = foi*(y['i',t]/popsize)
        dS = st.binom.rvs(n=int(y['s',t]), p=p_infection, size=1)[0] # [0] since returns an array
        dI = st.binom.rvs(n=int(y['i',t]), p=1.0/duration, size=1)[0]
        S = y['s',t] - dS
        I = y['i',t] + dS - dI
        R = y['r',t] + dI
        y[t+1] = [S, I, R] # TODO: allow opposite indices
    
    return y


def make_SIR(noise=0.0, optimum='min', verbose=True):
    pass
    
#    default_y = blowfly_sim(pars=default_blowfly_pars, initpop=None, npts=None)
#    default_stats = blowfly_statistics(default_y)
#    
#    def blowfly_err(pars):
#        y = blowfly_sim(pars=pars, initpop=None, npts=None)
#        stats = blowfly_statistics(y)
#        mismatch = stats['cumdist'] - default_stats['cumdist']
#        err = pl.sqrt(pl.mean(mismatch**2)) # Calculate RMSE between predicted and actual CDF
#        err = om.problem_suite.addnoise(err, noise)
#        if optimum == 'max':
#            err = -err
#        return err
#    
#    func = lambda pars: blowfly_err(pars) # Create the actual function
#    
#    if verbose:
#        print("Created blowfly function with noise=%s" % (noise))
#        print('Suggested starting point: [20,0.5]')
#        print('Suggested limits: r ~ [0,80] and σ ~ [0,2]')
#        print('Optimal solution: %s≈0.2 near %s' % (optimum, str(default_blowfly_pars)))
#    return func


def plot_SIR(pars=None, popsize=None, npts=None, fig=None):
    y = SIR_sim(pars=pars, popsize=popsize, npts=npts)
    x = pl.arange(len(y))
    
    # Allow points to be added to an existing figure
    if fig is None: fig = pl.figure()
    if len(fig.axes)<1: ax = pl.subplot(1,1,1)
    else:               ax = fig.axes[0]
        
    ax.plot(x, y['s'], marker='o', lw=2, label='Susceptible') # TODO: row slices?!
    ax.plot(x, y['i'], marker='o', lw=2, label='Infectious')
    ax.plot(x, y['r'], marker='o', lw=2, label='Recovered')
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of people')
#    ax.set_title('SIR simulation with r=%0.2f, σ=%0.2f' % (pars[0], pars[1]))
    ax.legend()

    output = sc.objdict({'x':x, 'y':y})
    return output