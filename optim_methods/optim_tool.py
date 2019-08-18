'''
Reimplementation of Dan K.'s OptimTool from dtk-tools.

The naming convention is to distinguish the module (optim_tool) from the function
(optimtool).

Version: 2019aug15
'''

import numpy as np
import sciris as sc
import scipy.special as sp # for calculation of mu_r
import scipy.stats as st

__all__ = ['optimtool', 'get_r']

def get_r(npars, vfrac):
    '''
    For a given number of parameters npars and volume fraction vfrac, calculate
    how large the radius should be to calculate the derivative. Used to set the 
    default for mu_r and sigma_r in optimtool.
    
    Parameters
    ----------
    npars : int
        Number of parameters (NB, expect >=2!)
    vfrac : float
        Fraction of volume

    Returns
    -------
    r : float
        Radius of the hypersphere
    
    Examples
    --------
    >>> import optim_methods as om
    >>> om.get_r(npars=4, vfrac=0.1)
    0.8381416784196289
    >>> om.get_r(npars=100, vfrac=0.1)
    0.3924137239465346
    
    See https://en.wikipedia.org/wiki/Volume_of_an_n-ball
    
    Version: 2019aug16 
    '''
    r = np.exp(1.0/npars * (np.log(vfrac) - sp.gammaln(npars/2.+1) + npars/2.*np.log(np.pi)))
    return r


def sample_hypersphere(mp, x, xmax, xmin, fittable):
    '''
    Sample points from a hypersphere. See tests/test_optimtool.py for usage example.
    '''
    
    # Initialize
    fitinds = sc.findinds(fittable) # The indices of the fittable parameters
    nfittable = len(fitinds) # How many fittable parameters
    npars = len(x) # Total number of parameters
    standard_normal = st.norm(loc=0, scale=1) # Initialize a standard normal distribution
    radius_normal = st.norm(loc=mp.mu_r, scale=mp.sigma_r) # And a scaled one
    
    # Calculate deviations
    deviations = np.zeros((mp.N, nfittable)) # Deviations from current center point
    for r in range(mp.N): # Loop over repeats
        sn_rvs = standard_normal.rvs(size=nfittable) # Sample from the standard distribution
        sn_nrm = np.linalg.norm(sn_rvs) # Calculate the norm of these samples
        radius = radius_normal.rvs() # Sample from the scaled distribution
        deviations[r,:] = radius/sn_nrm*sn_rvs # Deviation is the scaled sample adjusted by the rescaled standard sample
    
    # Calculate parameter samples
    samples = np.zeros((mp.N, npars)) # New samples
    for p in range(npars): # Loop over all parameters
        if fittable[p]:
            ind = sc.findinds(fitinds==p)[0] # Convert the parameter index back to the fittable parameter index
            delta = deviations[:,ind] * (xmax[p] - xmin[p]) # Scale the deviation by the allowed parameter range
        else:
            delta = 0 # If not fittable, set to zer
        samples[:,p] = x[p] + delta # Set new parameter value
    
    # Clamp
    for p in range(npars):
        samples[:,p] = np.minimum(xmax[p], np.maximum(xmin[p], samples[:,p])) # Ensure all samples are within range
    
    return samples


def optimtool(func, x, xmin=None, xmax=None, metapars=None, verbose=2):
    '''
    Reimplementation of the original dtk-tools OptimTool function:
        dtk-tools/calibtool/algorithms/OptimTool.py
    
    The algorithm has two basic parts. In the first part, it samples from a hyperdoughnut
    around the current point. In the second part, it takes this hypershell and tries to fit
    a hyperplane to it, and if it "succeeds" (the r-squared value is high enough),
    it will step in the uphill direction. Otherwise, it just takes the best point.
    
    Version: 2019aug16
    '''
    
    # Define defaults
    mp = sc.objdict({
            'mu_r':    get_r(),
            'sigma_r': get_r()/10,
            'N':    1,
            'center_repeats': 1,
            })
    
    
    
    
    
    def ascent():
        pass
    
    # Placeholder output
    output = sc.objdict()
    output['x'] = [] # Parameters
    output['fval'] = np.nan
    output['exitreason'] = 'n/a'
    output['details'] = sc.objdict()
    output['details']['fvals'] = [] # Function evaluations
    output['details']['xvals'] = []
    return output
