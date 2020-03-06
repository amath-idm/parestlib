'''
Density-weighted iterative threshold sampling.

The naming convention is to distinguish the module (shell_step) from the function
(shellstep).

Basic usage is:
    
    import parestlib as pe
    output = pe.binnts(func, x, xmin, xmax)

Version: 2020mar01
'''

import numpy as np
import sciris as sc
import scipy.stats as st

__all__ = ['BINNTS', 'binnts']


class BINNTS(sc.prettyobj):
    '''
    Class to implement density-weighted iterative threshold sampling.
    
    Version: 2020feb29
    '''
    
    def __init__(self, func, x, xmin, xmax, npoints=None, acceptance=None, nbootstrap=None, maxiters=None, optimum=None, func_args=None, verbose=None):
        
        # Handle required arguments
        self.func = func
        self.x    = np.array(x)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        
        # Handle optional arguments
        self.npoints     = npoints     if npoints    is not None else 100
        self.acceptance  = acceptance  if acceptance is not None else 0.5
        self.nbootstrap  = nbootstrap  if nbootstrap is not None else 10
        self.maxiters    = maxiters    if maxiters   is not None else 50 
        self.optimum     = optimum     if optimum    is not None else 'min'
        self.func_args   = func_args   if func_args  is not None else {}
        self.verbose     = verbose     if verbose    is not None else 2
        
        # Set up results
        self.ncandidates  = int(self.npoints/self.acceptance)
        self.iteration    = 0
        self.npars        = len(x) # Number of parameters being fit
        self.npriorpars   = 4 # Number of parameters in the prior distribution
        self.priorpars    = np.zeros((self.npars, self.npriorpars)) # Each parameter is defined by a 4-metaparameter prior distribution
        self.samples      = np.zeros((self.npoints, self.npars)) # Array of parameter values
        self.results      = np.zeros(self.npoints) # For storing the results object (see optimize())
        self.allpriorpars = [] # For storing history of the prior-distribution parameters
        self.allsamples   = [] # For storing all points
        self.allresults   = [] # For storing all results
        
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
        return
    
    
    def draw_samples(self, init=False): # TODO: refactor discrepancy between points and samples and candidates
        ''' Choose samples from the (current) prior distribution '''
        if init: # Initial samples
            for p in range(self.npars): # Loop over the parameters
                self.samples[:,p] = self.beta_rvs(pars=self.priorpars[p,:], n=self.npoints)
            return
        else: 
            candidates = np.zeros((self.ncandidates, self.npars))
            for p in range(self.npars): # Loop over the parameters
                candidates[:,p] = self.beta_rvs(pars=self.priorpars[p,:], n=self.ncandidates)
            return candidates
    
    
    def evaluate(self):
        ''' Actually evaluate the objective function '''
        for s,sample in enumerate(self.samples): # TODO: parallelize
            self.results[s] = self.func(sample, **self.func_args) # This is the time-consuming step!!
        self.allresults.append(sc.dcp(self.results))
        return
    
    
    # def step(self):
    #     ''' Calculate new samples based on the current samples and matching results '''
    
    #     # Calculate ordinary least squares fit of sample parameters on results
    #     if self.optimum == 'max': results = -self.results # Flip the sign if we're using the maximum
    #     else:                     results =  self.results # Default, just use the stored results
        
    #     new_samples = self.draw_samples()
    #     distances = np.zeros((self.npoints, self.nsamples))
    #     # Compute distances
        
        
    #     return self.samples
    
    
    def make_surfaces(self):
        ''' Create the bootstrapped surfaces '''
        self.bs_surfaces = np.zeros((self.nbootstrap, len(self.samples), self.npars+1))
        for b in range(self.nbootstrap):
            bs_samples = np.random.randint(0, len(self.samples), len(self.samples)) # TODO should be able to use npoints or nsamples?!
            for p in range(self.npars):
                self.bs_surfaces[b,:,p] = self.samples[bs_samples, p]
            self.bs_surfaces[b,:,-1] = self.results[bs_samples]
        
        return
    
    
    def optimize(self):
        ''' Actually perform an optimization '''
        self.initialize_priors() # Initialize
        self.draw_samples(init=True)
        self.evaluate() # Evaluate the objective function
        for i in range(self.maxiters): # Iterate
            if self.verbose>=1: print(f'Step {i+1} of {self.maxiters}')
            self.make_surfaces() # Calculate the bootstrapped surface of nearest neighbors
            self.draw_samples()
            self.estimate_samples()
            self.evaluate()
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
    dwest = BINNTS(*args, **kwargs) # Create class instance
    output = dwest.optimize() # Run the optimization
    return output
    
    
    
