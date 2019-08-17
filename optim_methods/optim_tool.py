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
    Sample points from a hypersphere
    '''
    
    import pylab as pl; pl.seed(857232)
    print('TEST NUMBER2: %s' % pl.rand())
    
    # Initialize
    npars = len(x)
    dt         = np.zeros((mp.N, npars))
    samples    = np.zeros((mp.N, npars))
    standard_normal = st.norm(loc=0, scale=1)
    radius_normal = st.norm(loc=mp.mu_r, scale=mp.sigma_r)
    
    # Calculate deltas
    for r in range(mp.N): # Loop over repeats
        sn_rvs = standard_normal.rvs(size=npars)
        sn_nrm = np.linalg.norm(sn_rvs)
        radius = radius_normal.rvs()
        dt[r,:] = radius/sn_nrm*sn_rvs
    
    print('MASHY')
    print(dt)

    # Calculate samples
    for p in range(npars): # Loop over parameters
        Xcen = x[p]
        if fittable[p]:
            delta = dt[:,p] * (xmax[p] - xmin[p])
        else:
            delta = 0
        samples[:,p] = Xcen + delta
    
    # Clamp
    for p in range(npars):
        samples[:,p] = np.minimum(xmax[p], np.maximum(xmin[p], samples[:,p]))
    
    return samples


def optimtool(func, x, xmin=None, xmax=None, metapars=None, verbose=2):
    '''
    Reimplementation of the original dtk-tools OptimTool function:
        dtk-tools/calibtool/algorithms/OptimTool.py
    
    The algorithm has two basic parts. In the first part, it samples from a hypershell
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
            '_lenpars': 1,
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
