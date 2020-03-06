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
            priordist = B.beta_pdf(bp, xvec)
            pl.plot(xvec, priordist, label=f'x={B.x[i]}, alpha={bp[0]:0.2f}, beta={bp[1]:0.2f}')
            pl.legend()
        pl.show()
        
    return prior_dist


def test_sampling(doplot=False):
    sc.heading('Create parameter samples')
    npoints = 1000
    nbins = 50
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, npoints=npoints)
    B.initialize_priors()
    samples = B.draw_samples()
    if doplot:
        pl.figure(figsize=figsize)
        for p in range(B.npars):
            pl.subplot(B.npars, 1, p+1)
            pl.hist(samples[:,p], bins=nbins)
            pl.title(f'Samples for x={B.x[p]}, β={B.priorpars[p,:]}')
        pl.show()
    return samples
    

def test_bootstrap(doplot=False):
    sc.heading('Bootstrapped parameter values')
    B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
    B.initialize_priors()
    B.draw_samples(init=True)
    B.evaluate()
    B.make_surfaces()
    if doplot:
        sf = 300 # Scale factor from value to point size for plotting
        x_ind = 0 # Which parameter corresponds to the x axis -- just for plotting
        y_ind = 1 # Ditto for y
        val_ind = B.npars # The value for sims is stored after the parameters
        pl.figure(figsize=figsize)
        for b in range(B.nbootstrap):
            pl.clf()
            this_bs = B.bs_surfaces[b]
            p1 = this_bs[:,x_ind]
            p2 = this_bs[:,y_ind]
            val = this_bs[:,val_ind]
            pl.scatter(p1, p2, s=val*sf)
            pl.title(f'Bootstrap {b+1} of {B.nbootstrap}, size ∝ error')
            pl.xlim([B.xmin[x_ind], B.xmax[x_ind]])
            pl.ylim([B.xmin[y_ind], B.xmax[y_ind]])
            pl.pause(0.2)
    return B.bs_surfaces


def test_estimation(doplot=False):
    sc.heading('Bootstrapped parameter values')
    
    


# def test_optimization():
#     sc.heading('Run an actual optimization')
#     B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
#     R = B.optimize()
#     return R


#%% Run as a script -- comment out lines to turn off tests
if __name__ == '__main__':
    sc.tic()
    B = test_creation()
    prior_dist = test_initial_prior(doplot=doplot)
    samples = test_sampling(doplot=doplot)
    bs_surfaces = test_bootstrap(doplot=doplot)
    # R = test_optimization(doplot=doplot)
    print('\n'*2)
    sc.toc()
    print('Done.')