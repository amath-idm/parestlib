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
    how large the radius should be.
    
    Parameters
    ----------
    npars : int
        Number of parameters
    vfrac : float
        Fraction of volume to be covered

    Returns
    -------
    r : float
        Radius of sphere that will cover the specified volume
    
    Example
    -------
    >>> get_r(npars=4, vfrac=0.1)
    0.8381416784196289
    
    Version: 2019aug15    
    '''
    r = np.exp(1.0/npars * (np.log(vfrac) - sp.gammaln(npars/2.+1) + npars/2.*np.log(np.pi)))
    return r


def optimtool(func, pars, verbose=2):
    
    
    
    def ascent():
        pass
    
    def sample_hypersphere():
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
