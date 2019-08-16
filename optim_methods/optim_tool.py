'''
Reimplementation of Dan K.'s OptimTool from dtk-tools.

The naming convention is to distinguish the module (optim_tool) from the function
(optimtool).

Version: 2019aug15
'''

import numpy as np
import sciris as sc
import scipy.special as sp # for calculation of mu_r

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
        Fraction of volume of the unit hypercube to cover by the hypersphere

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
    
    Version: 2019aug15    
    '''
    r = (np.pi**(npars/2.0)/sp.gammaln(npars/2.+1)*vfrac)**(1.0/npars) # Original: r = np.exp(1.0/npars * (np.log(vfrac) - sp.gammaln(npars/2.+1) + npars/2.*np.log(np.pi)))
    return r


def optimtool(func, x, metapars=None, verbose=2):
    '''
    Reimplementation of the original dtk-tools OptimTool function:
        dtk-tools/calibtool/algorithms/OptimTool.py
    
    The algorithm has two basic parts. In the first part, it samples from a hypershell
    around the current point. In the second part, it takes this hypershell and tries to fit
    a hyperplane to it, and if it "succeeds" (the r-squared value is high enough),
    it will step in the uphill direction. Otherwise, it just takes the best point.
    
    Version: 2019aug16
    '''
    
    def sample_hypersphere():
        pass
    
    
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
