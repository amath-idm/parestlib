'''
Tests of the utilies for the model.
'''

#%% Imports and settings
import pytest
import pylab as pl
import sciris as sc
import parestlib as pe


#%% Define the parameters
doplot    = True
figsize   = (18,12)
eqfigsize = (18,18)
x         = [0.2, 0.5]
xmin      = [0, 0]
xmax      = [1, 1]

#%% Define the tests


def test_distances(doplot=False):
    npoints = 1000
    nsamples = 2
    npars = 2
    test = pl.rand(nsamples, npars)
    train = pl.rand(npoints, npars)
    t1 = sc.tic()
    distances = pe.scaled_norm(test=test, train=train)
    t2 = sc.toc(t1, output=True)
    timestr = f'time = {t2*1e3:0.2f} ms'
    print(timestr)
    
    # Test a shape mismatch
    with pytest.raises(ValueError):
        pe.scaled_norm(test=pl.rand(7), train=pl.rand(7,4)) # Should be 4, not 7
        
    if doplot:
        x_ind = 0
        y_ind = 1
        offset = 0.009
        pl.figure(figsize=eqfigsize)
        sc.parulacolormap(apply=True) # or pl.set_map('parula')
        for pt in range(2):
            markers = ['<','>']
            markersize = 50
            bigmarker = 200
            pl.scatter(train[:,x_ind]+offset*pt, train[:,y_ind], s=markersize, c=distances[pt], marker=markers[pt], label=f'Samples {pt+1}')
            pl.scatter(test[pt][0], test[pt][1], s=bigmarker, c='k', marker=markers[pt], label=f'Origin {pt+1}')
        pl.xlabel('Parameter 1')
        pl.ylabel('Parameter 2')
        pl.title(f'Distance calculations (color ∝ distance); {timestr}')
        pl.legend()
        pl.colorbar()
        pl.axis('square')
    return distances


def test_estimates(doplot=False, verbose=False):
    
    # Set input data parameters
    ntrain = 200
    ntest  = 50
    npars  = 2
    noise  = 0.5
    seed   = 1
    
    # Set algorithm parameters
    k          = 3
    nbootstrap = 10
    weighted   = 1
    
    # Set up training and test arrays
    pl.seed(seed)
    train_arr = pl.rand(ntrain, npars)
    train_vals = pl.sqrt(((train_arr-0.5)**2).sum(axis=1)) + noise*pl.rand(ntrain) # Distance from center
    test_arr = pl.rand(ntest, npars)
    
    # Calculate the estimates
    t1 = sc.tic()
    test_vals = pe.bootknn(test=test_arr, train=train_arr, values=train_vals, k=k, nbootstrap=nbootstrap, weighted=weighted) 
    t2 = sc.toc(t1, output=True)
    timestr = f'time = {t2*1e3:0.2f} ms'
    print(timestr)
    
    if doplot:
        # Setup
        xind = 0
        yind = 1
        offset = 0.015
        cmap = 'parula'
        x_off = offset*pl.array([0, -1, 0, 1, 0]) # Offsets in the x direction
        y_off = offset*pl.array([-1, 0, 0, 0, 1]) # Offsets in the y direction
        train_args = dict(marker='o', s=50)
        test_args  = dict(marker='s', s=80)
        minval = min(train_vals.min(), test_vals.array.min())
        maxval = min(train_vals.max(), test_vals.array.max())
        train_colors = sc.arraycolors(train_vals,      cmap=cmap, minval=minval, maxval=maxval)
        test_colors  = sc.arraycolors(test_vals.array, cmap=cmap, minval=minval, maxval=maxval)
        
        # Make the figure
        pl.figure(figsize=eqfigsize)
        pl.scatter(train_arr[:,xind], train_arr[:,yind], c=train_colors, **train_args, label='Training')
        
        # Plot the data
        for q in range(ntest):
            for i in range(5):
                label = 'Predicted' if i==0 and q==0 else None # To avoid appearing multiple times
                x = test_arr[q,xind]+x_off[i]
                y = test_arr[q,yind]+y_off[i]
                v = test_vals.array[i,q]
                c = test_colors[i,q]
                pl.scatter(x, y, c=[c], **test_args, label=label)
                if verbose:
                    print(f'i={i}, q={q}, x={x:0.3f}, y={y:0.3f}, v={v:0.3f}, c={c}')
                    pl.pause(0.3)
        
        pl.xlabel('Parameter 1')
        pl.ylabel('Parameter 2')
        pl.title(f'Parameter estimates; {timestr}')
        pl.legend()
        pl.set_cmap(cmap)
        pl.clim((minval, maxval))
        pl.axis('square')
        pl.colorbar()
    
    return test_vals


def test_beta_fit(doplot=False):
    n = 100
    data = pl.randn(n)*0.3+0.7
    pars = pe.beta_fit(data)
    if doplot:
        pl.hist(data, bins=20, density=True, alpha=0.6, color='g')
        xmin, xmax = pl.xlim()
        xvec = pl.linspace(xmin, xmax, 100)
        p = pe.beta_pdf(pars, xvec)
        pl.plot(xvec, p, 'k', linewidth=2)
        pl.title(f"Fit results: mu={pars[0]:.2f}, std={pars[1]:.2f}")
    return pars



#%% Run as a script -- comment out lines to turn off tests
    
if __name__ == '__main__':
    sc.tic()
    distances = test_distances(doplot=doplot)
    estimates = test_estimates(doplot=doplot)
    pars = test_beta_fit(doplot=doplot)
    print('\n'*2)
    sc.toc()
    print('Done.')

