'''
Tests of the BINNTS algorithm.
'''

#%% Imports and settings
import pytest
import pylab as pl
import sciris as sc
import parestlib as pe


#%% Define the parameters
doplot    = True
figsize   = (18,12)
x         = [0.2, 0.5]
xmin      = [0, 0]
xmax      = [1, 1]
binnts_pars = {}
binnts_pars['nsamples']     = 50
binnts_pars['acceptance']  = 0.5
binnts_pars['nbootstrap']  = 10
binnts_pars['maxiters']    = 5 

def objective(x):
    ''' Example objective function '''
    return pl.norm(x) # TODO: add optional noise


#%% Define the tests

def test_creation():
    sc.heading('Create class')
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax)
    assert B.iteration == 0 # Only one of various things that could be tested
    output = B.func(x)
    print(f'Default output is: {output}')
    return B
    

def test_initial_prior(doplot=False):
    sc.heading('Create prior distributions')
    width = 0.1
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax)
    prior_dist_u = B.initialize_priors(prior='uniform')
    print(f'Uniform prior distribution is:\n{prior_dist_u}')
    B.initialize_priors(width=width)
    prior_dist = B.priorpars
    print(f'"Best" prior distribution for x={B.x} is:\n{prior_dist}')
    
    # Check that not found distributions crash
    with pytest.raises(NotImplementedError):
        B.initialize_priors(prior='something_mistyped')
    
    # Optionally plot
    if doplot:
        pl.figure(figsize=figsize)
        for i in range(B.npars):
            bp = prior_dist[i]
            xvec = pl.linspace(B.xmin[i], B.xmax[i])
            priordist = pe.beta_pdf(bp, xvec)
            pl.plot(xvec, priordist, label=f'x={B.x[i]}, alpha={bp[0]:0.2f}, beta={bp[1]:0.2f}')
            pl.legend()
        pl.show()
        
    return prior_dist


def test_sampling(doplot=False):
    sc.heading('Create parameter samples')
    nsamples = 1000
    nbins = 50
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, nsamples=nsamples)
    B.initialize_priors()
    B.draw_initial_samples()
    if doplot:
        pl.figure(figsize=figsize)
        for p in range(B.npars):
            pl.subplot(B.npars, 1, p+1)
            pl.hist(B.samples[:,p], bins=nbins)
            pl.title(f'Samples for x={B.x[p]}, β={B.priorpars[p,:]}')
        pl.show()
    return B.samples
    

def test_optimization(doplot=False):
    sc.heading('Run an actual optimization')
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
    R = B.optimize()
    return R


#%% Run as a script -- comment out lines to turn off tests
if __name__ == '__main__':
    sc.tic()
    B = test_creation()
    prior_dist = test_initial_prior(doplot=doplot)
    samples = test_sampling(doplot=doplot)
    # bs_pars, bs_vals = test_bootstrap(doplot=doplot)
    R = test_optimization(doplot=doplot)
    print('\n'*2)
    sc.toc()
    print('Done.')