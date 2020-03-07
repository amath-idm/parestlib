'''
Density-weighted iterative threshold sampling.

The naming convention is to distinguish the module (shell_step) from the function
(shellstep).

Basic usage is:
    
    import parestlib as pe
    output = pe.binnts(func, x, xmin, xmax)

Version: 2020mar06
'''

import numpy as np
import sciris as sc
import scipy.stats as st

__all__ = ['calculate_distances', 'BINNTS', 'binnts']


def calculate_distances(point, arr, quantiles=None):
    '''
    Calculation of distances -- was going to use Numba but plenty fast without.
    Before calculating distances, normalize each dimension to have the same "scale"
    (default: interquartile range).
    '''
    
    # Handle inputs
    if quantiles is None:
        quantiles = [0.25, 0.75] # Default quantiles to compute scale from
    npars = len(point)
    npoints = len(arr)
    if arr.shape != (npoints, npars):
        raise ValueError(f'Array shape appears to be incorrect: {arr.shape} vs {(npoints, npars)}')
    
    # Copy; otherwise, these get modified in place
    point = sc.dcp(point)
    arr = sc.dcp(arr)
    
    # Normalize
    for p in range(npars):
        scale = np.diff(np.quantile(arr[:,p], quantiles))
        arr[:,p] /= scale # Transform to be of comparable scale
        point[p] /= scale # For point too
        
    # The actual calculation
    distances = np.linalg.norm(arr - point, axis=1)
    return distances


class BINNTS(sc.prettyobj):
    '''
    Class to implement density-weighted iterative threshold sampling.
    '''
    
    def __init__(self, func, x, xmin, xmax, neighbors=None, npoints=None, 
                 acceptance=None, nbootstrap=None, nfolds=None, leaveout=None,
                 maxiters=None, optimum=None, func_args=None, verbose=None):
        
        # Handle required arguments
        self.func = func
        self.x    = np.array(x)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        
        # Handle optional arguments
        self.neighbors   = neighbors   if neighbors  is not None else 3
        self.npoints     = npoints     if npoints    is not None else 100
        self.acceptance  = acceptance  if acceptance is not None else 0.5
        self.nbootstrap  = nbootstrap  if nbootstrap is not None else 10
        # self.nfolds      = nfolds      if nfolds     is not None else 5
        # self.leaveout    = leaveout    if leaveout   is not None else 0.2
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
        self.candidates   = np.zeros((self.ncandidates, self.npars)) # Array of candidate samples
        self.results      = np.zeros(self.npoints) # For storing the results object (see optimize())
        self.allpriorpars = np.zeros((self.maxiters, self.npars, self.npriorpars)) # For storing history of the prior-distribution parameters
        self.allsamples   = np.zeros((0, self.npars), dtype=float) # For storing all points
        self.allresults   = np.zeros(0, dtype=float) # For storing all results
        
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
        ''' Actually evaluate the objective function '''
        for s,sample in enumerate(self.samples): # TODO: parallelize
            self.results[s] = self.func(sample, **self.func_args) # This is the time-consuming step!!
        self.allsamples = np.concatenate([self.allsamples, self.samples])
        self.allresults = np.concatenate([self.allresults, self.results])
        return
    
    
    def make_surfaces(self):
        ''' Create the bootstrapped surfaces '''
        
        # Create surfaces
        self.bs_pars = np.zeros((self.nbootstrap, len(self.allsamples), self.npars))
        self.bs_vals = np.zeros((self.nbootstrap, len(self.allsamples)))
        for b in range(self.nbootstrap):
            bs_samples = np.random.randint(0, len(self.allsamples), len(self.allsamples)) # TODO should be able to use npoints or nsamples?!
            for p in range(self.npars):
                self.bs_pars[b,:,p] = self.allsamples[bs_samples, p]
            self.bs_vals[b,:] = self.allresults[bs_samples]
        
        # Evaluate surfaces and choose number of neighbors
        # folds = []
        # for f in range(self.nfolds):
        #     n_inds = len(self.samples)
        #     all_inds = np.random.randint(0, n_inds)
            # in_inds = all_inds[]
        
        return
    
    
    def estimate_samples(self):
        ''' Calculate an estimated value for each of the candidate points '''
        
        # Calculate distances
        distances = np.zeros((self.nbootstrap, self.ncandidates, len(self.allsamples))) # Matrix of all distances
        for b in range(self.nbootstrap):
            bs_pars = self.bs_pars[b,:,:] # e.g. 100 points with 5 parameter values
            for c in range(self.ncandidates):
                candidate = self.candidates[c,:]
                distances[b,c,:] = calculate_distances(candidate, bs_pars)
        
        # Calculate estimates
        estimates = #
        variances = #
        
        # Choose best points
        ...
                
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
    
    
    
