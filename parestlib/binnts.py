'''
Bootstrapped iterative nearest-neighbor threshold sampling (BINNTS.

Basic usage is:
    
    import parestlib as pe
    output = pe.binnts(func, x, xmin, xmax)

Version: 2020mar07
'''

import numpy as np
import sciris as sc
import scipy.stats as st
from . import utils as ut

__all__ = ['BINNTS', 'binnts']


class BINNTS(sc.prettyobj):
    '''
    Class to implement density-weighted iterative threshold sampling.
    '''
    
    def __init__(self, func, x, xmin, xmax, neighbors=None, npoints=None, 
                 acceptance=None, k=None, nbootstrap=None, quantile=None,
                 maxiters=None, optimum=None, func_args=None, verbose=None,
                 parallel_args=None, parallelize=None):
        
        # Handle required arguments
        self.func = func
        self.x    = np.array(x)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        
        # Handle optional arguments
        self.neighbors     = neighbors     if neighbors     is not None else 3
        self.npoints       = npoints       if npoints       is not None else 100
        self.acceptance    = acceptance    if acceptance    is not None else 0.5
        self.k             = k             if k             is not None else 3
        self.nbootstrap    = nbootstrap    if nbootstrap    is not None else 10
        self.nbootstrap    = nbootstrap    if nbootstrap    is not None else 10
        self.maxiters      = maxiters      if maxiters      is not None else 20
        self.optimum       = optimum       if optimum       is not None else 'min'
        self.func_args     = func_args     if func_args     is not None else {}
        self.verbose       = verbose       if verbose       is not None else 2
        self.parallel_args = parallel_args if parallel_args is not None else {}
        self.parallelize   = parallelize   if parallelize   is not None else False
        
        # Set up results
        self.ncandidates  = int(self.npoints/self.acceptance)
        self.iteration    = 0
        self.npars        = len(x) # Number of parameters being fit
        self.npriorpars   = 4 # Number of parameters in the prior distribution
        self.priorpars    = np.zeros((self.npars, self.npriorpars)) # Each parameter is defined by a 4-metaparameter prior distribution
        self.samples      = np.zeros((self.npoints, self.npars)) # Array of parameter values
        self.candidates   = np.zeros((self.ncandidates, self.npars)) # Array of candidate samples
        self.values      = np.zeros(self.npoints) # For storing the values object (see optimize())
        self.allpriorpars = np.zeros((self.maxiters, self.npars, self.npriorpars)) # For storing history of the prior-distribution parameters
        self.allsamples   = np.zeros((0, self.npars), dtype=float) # For storing all points
        self.allvalues   = np.zeros(0, dtype=float) # For storing all values
        
        return
    
    
    @staticmethod
    def beta_pdf(pars, xvec):
        ''' Shortcut to the scipy.stats beta PDF function -- not used currently, but nice to have '''
        if len(pars) != 4:
            raise Exception(f'Beta distribution parameters must have length 4, not {len(pars)}')
        pdf = st.beta.pdf(x=xvec, a=pars[0], b=pars[1], loc=pars[2], scale=pars[3])
        return pdf
    
    
    @staticmethod
    def beta_rvs(pars, n):
        ''' Shortcut to the scipy.stats beta random variates function '''
        if len(pars) != 4:
            raise Exception(f'Beta distribution parameters must have length 4, not {len(pars)}')
        rvs = st.beta.rvs(a=pars[0], b=pars[1], loc=pars[2], scale=pars[3], size=n)
        return rvs
    
    
    def initialize_priors(self, prior='best', width=0.5):
        ''' Create the initial prior distributions '''
        if isinstance(prior, type(np.array([]))):
            prior_shape = (self.npars, self.npriorpars)
            if prior.shape != prior_shape:
                raise Exception(f'Shape of prior is wrong: {prior.shape} instead of {prior_shape}')
            self.priorpars = prior # Use supplied parameters directly
            return
        for p in range(self.npars): # Loop over the parameters
            xloc   = self.xmin[p]
            xscale = self.xmax[p] - xloc
            if prior == 'uniform': # Just use a uniform prior from min to max
                alpha  = 1
                beta   = 1
            elif prior == 'best': # Use the best guess parameter value to create the distribution
                best = (self.x[p] - xloc)/xscale # Normalize best guess
                swap = False # Check if values need to be swapped since the distribution is not symmetric about 0.5
                if best > 0.5:
                    best = 1 - best
                    swap = True
                alpha = 1.0/width
                beta = alpha*(1.0-best)/best # Use best as the mean
                if swap:
                    alpha, beta = beta, alpha # If we swapped earlier, swap back now
            else:
                raise NotImplementedError('Currently, only "uniform" and "best" priors are supported')
            self.priorpars[p,:] = [alpha, beta, xloc, xscale]
        self.allpriorpars[self.iteration,:,:] = self.priorpars
        return
    
    
    def draw_samples(self, init=False): # TODO: refactor discrepancy between points and samples and candidates
        ''' Choose samples from the (current) prior distribution '''
        for p in range(self.npars): # Loop over the parameters
            if init: # Initial samples
                self.samples[:,p] = self.beta_rvs(pars=self.priorpars[p,:], n=self.npoints)
            else: 
                self.candidates[:,p] = self.beta_rvs(pars=self.priorpars[p,:], n=self.ncandidates)
        return
    
    
    def evaluate(self):
        ''' Actually evaluate the objective function -- copied from shell_step.py '''
        if not self.parallelize:
            for s,sample in enumerate(self.samples):
                self.values[s] = self.func(sample, **self.func_args) # This is the time-consuming step!!
        else:
            valueslist = sc.parallelize(self.func, iterarg=self.samples, kwargs=self.func_args, **self.parallel_args)
            self.values = np.array(valueslist, dtype=float)
        self.allsamples = np.concatenate([self.allsamples, self.samples])
        self.allvalues = np.concatenate([self.allvalues, self.values])
        return
    
    
    def choose_samples(self, which='low'):
        ''' Calculate an estimated value for each of the candidate points '''
    
        # Calculate estimates
        output = ut.bootknn(test=self.candidates, training=self.allsamples, values=self.allvalues)
        estimates = output[which]
        
        # Choose best points
        ...
                
        return
    
    
    def optimize(self):
        ''' Actually perform an optimization '''
        self.initialize_priors() # Initialize prior distributions
        self.draw_initial_samples() # Draw initial parameter samples
        self.evaluate_samples() # Evaluate the objective function at each sample point
        for i in range(self.maxiters): # Iterate
            if self.verbose>=1: print(f'Step {i+1} of {self.maxiters}')
            self.draw_candidates() # Draw a new set of candidate points
            self.choose_samples() # Use DWKNN
            self.evaluate_samples()
            self.check_fit()
            self.refit_priors()
        
        # Create output structure
        output = sc.objdict()
        output['x'] = self.x # Parameters
        output['fval'] = self.func(self.x, **self.func_args) # TODO: consider placing this elsewhere
        output['exitreason'] = f'Reached maximum iteration limit ({self.maxiters})' # Stopping conditions not really implemented yet
        output['obj'] = self # Just return the entire original object
        return output
            


def binnts(*args, **kwargs):
    '''
    Wrapper for BINNTS class
    '''
    B = BINNTS(*args, **kwargs) # Create class instance
    output = B.optimize() # Run the optimization
    return output
    
    
    
