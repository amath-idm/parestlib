'''
Numerical utilities; currently used for BINNTs.
'''

import numpy as np
import sciris as sc
import scipy.stats as st


__all__ = ['beta_pdf', 'beta_rvs', 'beta_fit', 'gof', 'scaled_norm', 'bootknn']


#%% Beta distribution helper functions

def beta_pdf(pars, xvec):
    ''' Shortcut to the scipy.stats beta PDF function -- not used currently, but nice to have '''
    if len(pars) != 4:
        raise Exception(f'Beta distribution parameters must have length 4, not {len(pars)}')
    pdf = st.beta.pdf(x=xvec, a=pars[0], b=pars[1], loc=pars[2], scale=pars[3])
    return pdf


def beta_rvs(pars, n):
    ''' Shortcut to the scipy.stats beta random variates function '''
    if len(pars) != 4:
        raise Exception(f'Beta distribution parameters must have length 4, not {len(pars)}')
    rvs = st.beta.rvs(a=pars[0], b=pars[1], loc=pars[2], scale=pars[3], size=n)
    return rvs


def beta_fit(data):
    ''' Fit a beta distribution to the data '''
    mu, std, a, b = st.beta.fit(data)
    pars = np.array([mu, std, a, b], dtype=float)
    return pars


def gof(actual, predicted, estimator='median fractional', use_mean=False, use_frac=True, use_squared=False, die=True, eps=1e-9):
    ''' Calculate the goodness of fit. Default estimator is mean fractional error. '''
    
    # Handle the estimator argument, if supplied
    if estimator is not None:
        
        # Handle default cases by setting input arguments
        if estimator == 'mean fractional':
            use_mean = True
            use_frac = True
        elif estimator == 'mean absolute':
            use_mean = True
            use_frac = False
        elif estimator == 'median fractional':
            use_mean = False
            use_frac = True
        elif estimator == 'median absolute':
            use_mean = False
            use_frac = False
        
        # Use sklearn
        else:
            try:
                import sklearn.metrics as sm
                sklearn_gof = getattr(sm, estimator) # Shortcut to e.g. sklearn.metrics.max_error
            except ImportError as E:
                raise ImportError(f'You must have sklearn >=0.22.2 installed: {str(E)}')
            except AttributeError:
                raise AttributeError(f'Estimator {estimator} is not available; see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter for options')
        
            output = sklearn_gof(actual, predicted)
            return output
    
    # Use default methods here
    mismatches = abs(actual - predicted)
    
    if use_squared:
        mismatches = mismatches**2
    
    if use_frac:
        if (actual<0).any() or (predicted<0).any():
            errormsg = 'WARNING: Calculating fractional errors for non-positive quantities is ill-advised!'
            if die:
                raise ValueError(errormsg)
            else:
                print(errormsg)
        else:
            mismatches /= (actual+eps)
    
    if use_mean:
        output = mismatches.mean()
    else:
        output = np.median(mismatches)
            
    return output
    


#%% Estimation functions

def scaled_norm(test, train, quantiles='IQR'):
    '''
    Calculation of distances between a test set of points and a training set of
    points -- was going to use Numba but plenty fast without.
    
    Before calculating distances, normalize each dimension to have the same "scale"
    (default: interquartile range).
    
    "test" can be a single point or an array of points.
    '''
    
    # Handle inputs
    if quantiles in [None, 'iqr', 'IQR']:
        quantiles = [0.25, 0.75] # Default quantiles to compute scale from
    elif not sc.checktype(quantiles, 'arraylike'):
        raise TypeError(f'Cound not understand quantiles {type(quantiles)}: should be "IQR" or array-like')
    
    # Copy; otherwise, these get modified in place
    test = sc.dcp(test)
    train = sc.dcp(train)
    
    # Dimension checking
    if test.ndim == 1:
        test = np.array([test]) # Ensure it's 2-dimensional
    
    ntest, npars = test.shape
    ntrain, npars2 = train.shape
    if npars != npars2:
        raise ValueError(f'Array shape appears to be incorrect: {npars2} should be {npars}')
    
    # Normalize
    for p in range(npars):
        scale = np.diff(np.quantile(train[:,p], quantiles))
        train[:,p] /= scale # Transform to be of comparable scale
        test[:,p] /= scale # For test points too
        
    # The actual calculation
    distances = np.zeros((ntest, ntrain))
    for i in range(ntest):
        distances[i,:] = np.linalg.norm(train - test[i,:], axis=1)
    
    if len(distances) == 1:
        distances = distances.flatten() # If we have only a single point, return a vector of distances
    
    return distances


def bootknn(test, train, values, k=3, nbootstrap=10, weighted=1, scale_quantiles=None, output_quantiles=None, outputkey='all'):
    '''
    Perform boostrapped distance-weighted k-nearest-neighbors estimation.
    
    Usage:
        output = pe.bootknn(test_points, train_points, train_values)
    
    Args:
        test (NxP float): Test set: N points in P-dimensional space for which the values need to be estimated
        train (MxP float): Training set: M pointsin P-dimensional space for which the values are known
        values (M float): Values to match the training data
        k (int): Number of nearest neighbors; default 3
        nbootstrap (int): Number of bootstrap iterations; default 10
        weighted (float): Whether or not neighbors should be weighted by distance; default 0, 1=50% weight to distance, 10=90% weight
        scale_quantiles (2 int): Pair of quantiles for bound estimation
        output_quantiles (2 int): Pair of quantiles for the low and high estimates of the output
        outputkey (str): Which quantity to output, can be 'best', 'min', etc.; default 'all', which returns the full object
    
    Returns:
        output (objdict): An object with best, low, and high estimates of the value at each test point
    '''
    
    # Handle inputs
    if scale_quantiles is None:
        scale_quantiles = [0.25, 0.75]
    if output_quantiles is None:
        output_quantiles = [0.25, 0.75]
        
    k = int(k)
    nbootstrap = int(nbootstrap) # Ensure it's the right
    if k < 1:
        print('Warning: the minimum value for k is 1; you supplied {k}')
        k = 1
    if nbootstrap < 1:
        print('Warning: the minimum value for nbootstrap is 1; you supplied {nbootstrap}')
        nbootstrap = 1
    
    # Calculate MxN distance array
    distances = scaled_norm(test, train, quantiles=scale_quantiles)
    
    # Array for holding all points # TODO: Numbafy if slow
    ntest = len(test)
    ntrain = len(train)
    est_arr = np.zeros((ntest, nbootstrap))
    for b in range(nbootstrap):
        if nbootstrap == 1:
            bs_inds = range(ntrain) # Just pick all indices
        else: # Do bootstrapping
            bs_inds = np.random.randint(0, ntrain, ntrain) # Bootstrapped indices of the training data
        bs_values = values[bs_inds]
        bs_distances = distances[:,bs_inds]
        for p in range(ntest):
            point_dists = bs_distances[p,:]
            order = point_dists.argsort() # Sort by distance # TODO: consider np.argpartition if slow
            neighbor_inds = order[:k] # Pick out the k-nearest neighbors
            neighbor_values = bs_values[neighbor_inds]
            if weighted:
                neighbor_dists = point_dists[neighbor_inds]
                neighbor_dists /= neighbor_dists.max() # Normalize
                closeness = 1.0/(neighbor_dists+1/weighted) # Scale by the 
                est = (neighbor_values*closeness).sum()/closeness.sum()
            else:
                est = neighbor_values.mean() # TODO: is it consistent to take the mean here but the median later?
            est_arr[p,b] = est
    
    # Calculate statistics
    quantiles = [0, output_quantiles[0], 0.5, output_quantiles[1], 1]
    stats = np.quantile(est_arr[:,:], quantiles, axis=1)
    
    # Construct output
    output = sc.objdict()
    output_keys = ['min', 'low', 'best', 'high', 'max']
    for k,key in enumerate(output_keys):
        output[key] = stats[k,:]
    output.quantiles = quantiles # Store the quantiles
    output.array = stats # Store the full array
    
    # Handle outputting a single quantity
    if outputkey not in [None, 'all']:
        if outputkey not in output_keys:
            raise KeyError(f'"which" must be one of {output_keys}')
        else:
            return output[outputkey]
    
    return output