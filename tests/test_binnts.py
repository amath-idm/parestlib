'''
Tests of the utilies for the model.
'''

#%% Imports and settings
import pytest
import pylab as pl
import sciris as sc
import parestlib as pe


#%% Define the parameters
doplot  = True
figsize = (18,12)
x       = [0.2, 0.5]
xmin    = [0, 0]
xmax    = [1, 1]
binnts_pars = {}
binnts_pars['npoints']     = 50
binnts_pars['acceptance']  = 0.5
binnts_pars['nbootstrap']  = 10
binnts_pars['maxiters']    = 5 



#%% Define the tests

def objective(x):
    ''' Example objective function '''
    return pl.norm(x) # TODO: add optional noise


def test_creation():
    sc.heading('Create class')
    D = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax)
    assert D.iteration == 0 # Only one of various things that could be tested
    output = D.func(x)
    print(f'Default output is: {output}')
    return D
    

def test_initial_prior(doplot=False):
    sc.heading('Create prior distributions')
    width = 0.1
    D = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax)
    prior_dist_u = D.initialize_priors(prior='uniform')
    print(f'Uniform prior distribution is:\n{prior_dist_u}')
    prior_dist = D.initialize_priors(width=width)
    print(f'"Best" prior distribution for x={D.x} is:\n{prior_dist}')
    
    # Check that not found distributions crash
    with pytest.raises(NotImplementedError):
        D.initialize_priors(prior='something_mistyped')
    
    # Optionally plot
    if doplot:
        pl.figure(figsize=figsize)
        for i in range(D.npars):
            bp = prior_dist[i]
            xvec = pl.linspace(D.xmin[i], D.xmax[i])
            priordist = D.beta_pdf(bp, xvec)
            pl.plot(xvec, priordist, label=f'x={D.x[i]}, alpha={bp[0]:0.2f}, beta={bp[1]:0.2f}')
            pl.legend()
        pl.show()
        
    return prior_dist


def test_sampling(doplot=False):
    sc.heading('Create parameter samples')
    npoints = 1000
    nbins = 50
    D = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, npoints=npoints)
    D.initialize_priors()
    samples = D.draw_samples()
    if doplot:
        pl.figure(figsize=figsize)
        for p in range(D.npars):
            pl.subplot(D.npars, 1, p+1)
            pl.hist(samples[:,p], bins=nbins)
            pl.title(f'Samples for x={D.x[p]}, β={D.priorpars[p,:]}')
        pl.show()
    return samples
    

def test_bootstrap(doplot=False):
    sc.heading('Run an actual optimization')
    D = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
    D.initialize_priors()
    D.draw_samples(init=True)
    D.evaluate()
    D.make_surfaces()
    if doplot:
        sf = 300 # Scale factor from value to point size for plotting
        pl.figure(figsize=figsize)
        for b in range(D.nbootstrap):
            this_bs = D.bs_surfaces[b]
            p1 = this_bs[:,0]
            p2 = this_bs[:,1]
            val = this_bs[:,2]
            pl.scatter(p1, p2, s=val*sf)
            pl.title(f'Bootstrap {b+1} of {D.nbootstrap}, size ∝ error')
            pl.pause(0.2)
    return D.bs_surfaces



# def test_optimization():
#     sc.heading('Run an actual optimization')
#     D = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
#     R = D.optimize()
#     return R


#%% Run as a script -- comment out lines to turn off tests
if __name__ == '__main__':
    sc.tic()
    # D = test_creation()
    # prior_dist = test_initial_prior(doplot=doplot)
    # samples = test_sampling(doplot=doplot)
    bs_surfaces = test_bootstrap(doplot=doplot)
    # R = test_optimization(doplot=doplot)
    print('\n'*2)
    sc.toc()
    print('Done.')