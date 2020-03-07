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
        pl.figure(figsize=figsize)
        for b in range(B.nbootstrap):
            pl.clf()
            this_bs = B.bs_pars[b]
            vals = B.bs_vals[b]
            p1 = this_bs[:,x_ind]
            p2 = this_bs[:,y_ind]
            pl.scatter(p1, p2, s=vals*sf)
            pl.title(f'Bootstrap {b+1} of {B.nbootstrap}, size ∝ error')
            pl.xlim([B.xmin[x_ind], B.xmax[x_ind]])
            pl.ylim([B.xmin[y_ind], B.xmax[y_ind]])
            pl.pause(0.2)
    return B.bs_pars, B.bs_vals


def test_distances(doplot=False):
    npoints = 1000
    nsamples = 2
    npars = 2
    test = pl.rand(nsamples, npars)
    training = pl.rand(npoints, npars)
    t1 = sc.tic()
    distances = pe.calculate_distances(test=test, training=training)
    t2 = sc.toc(t1, output=True)
    timestr = f'time = {t2*1e3:0.2f} ms'
    print(timestr)
    
    # Test a shape mismatch
    with pytest.raises(ValueError):
        pe.calculate_distances(test=pl.rand(7), training=pl.rand(7,4)) # Should be 4, not 7
        
    if doplot:
        x_ind = 0
        y_ind = 1
        offset = 0.009
        pl.figure(figsize=figsize)
        sc.parulacolormap(apply=True)
        for pt in range(2):
            markers = ['<','>']
            markersize = 50
            bigmarker = 200
            pl.scatter(training[:,x_ind]+offset*pt, training[:,y_ind], s=markersize, c=distances[pt], marker=markers[pt], label=f'Samples {pt+1}')
            pl.scatter(test[pt][0], test[pt][1], s=bigmarker, c='k', marker=markers[pt], label=f'Origin {pt+1}')
        pl.xlabel('Parameter 1')
        pl.ylabel('Parameter 2')
        pl.title(f'Distance calculations (color ∝ distance); {timestr}')
        pl.legend()
        pl.colorbar()
        pl.axis('square')
    return distances


def test_estimates(doplot=False, plot_training=False):
    ntraining = 100
    ntest = 50
    nbootstrap = 10
    k = 3
    npars = 2
    noise = 0.2
    training_arr = pl.rand(ntraining, npars)
    training_vals = pl.sqrt(((training_arr-0.5)**2).sum(axis=1)) + noise*pl.rand(ntraining) # Distance from center
    
    
    test_arr = pl.rand(ntest, npars)
    test_vals = pe.knn(test=test_arr, training=training_arr, values=training_vals, k=k, nbootstrap=nbootstrap)
    
    if doplot:
        xind = 0
        yind = 1
        training = dict(marker='o', s=50)
        test     = dict(marker='*', s=100)
        pl.figure(figsize=figsize)
        pl.scatter(training_arr[:,xind], training_arr[:,yind], c=training_vals, **training)
        pl.scatter(test_arr[:,xind],     test_arr[:,yind],     c=test_vals,     **test)
    
    
    # t1 = sc.tic()
    # distances = pe.calculate_distances(point, arr)
    # t2 = sc.toc(t1, output=True)
    # timestr = f'time = {t2*1e3:0.2f} ms'
    # print(timestr)
    
    # # Test a shape mismatch
    # with pytest.raises(ValueError):
    #     pe.calculate_distances(point=pl.rand(7), arr=pl.rand(7,4)) # Should be 4, not 7
        
    # if doplot:
    #     x_ind = 0
    #     y_ind = 1
    #     pl.figure(figsize=figsize)
    #     pl.scatter(arr[:,x_ind], arr[:,y_ind], c=distances, label='Samples')
    #     pl.scatter(point[0], point[1], s=200, c=[[0]*3], label='Origin')
    #     pl.xlabel('Parameter 1')
    #     pl.ylabel('Parameter 2')
    #     pl.title(f'Distance calculations (color ∝ distance); {timestr}')
    #     pl.legend()
    # return distances
    


# def test_estimation(doplot=False):
#     sc.heading('Estimated parameter values')
#     B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
#     B.initialize_priors()
#     B.draw_samples(init=True)
#     B.evaluate()
#     B.make_surfaces()
#     B.estimate_samples()
#     return B.bs_surfaces


# def test_optimization():
#     sc.heading('Run an actual optimization')
#     B = pe.BINNTS(func=objective, x=x, xmin=xmin, xmax=xmax, **binnts_pars)
#     R = B.optimize()
#     return R


#%% Run as a script -- comment out lines to turn off tests
if __name__ == '__main__':
    sc.tic()
    # B = test_creation()
    # prior_dist = test_initial_prior(doplot=doplot)
    # samples = test_sampling(doplot=doplot)
    # bs_pars, bs_vals = test_bootstrap(doplot=doplot)
    # distances = test_distances(doplot=doplot)
    estimates = test_estimates(doplot=doplot)
    # R = test_optimization(doplot=doplot)
    print('\n'*2)
    sc.toc()
    print('Done.')