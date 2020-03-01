'''
Density-weighted iterative threshold sampling.

The naming convention is to distinguish the module (shell_step) from the function
(shellstep).

Basic usage is:
    
    import parestlib as pe
    output = pe.dwits(func, x, xmin, xmax)

Version: 2020mar01
'''

import numpy as np
import sciris as sc
import scipy.stats as st

__all__ = ['DWITS', 'dwits']


class DWITS(sc.prettyobj):
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
        self.iteration = 0
        self.npars     = len(x) # Number of parameters being fit
        self.nbetapars = 4 # Number of parameters in the beta distribution
        self.betapars = np.zeros((self.npars, self.nbetapars)) # Each parameter is defined by a 4-metaparameter beta distribution
        self.samples  = np.zeros((self.npoints, self.npars)) # Array of parameter values
        self.results  = np.zeros(self.npoints) # For storing the results object (see optimize())
        self.allbetapars = [] # For storing history of the beta-distribution parameters
        self.allsamples  = [] # For storing all points
        self.allresults  = [] # For storing all results
        
        return
    
    
    def evaluate(self):
        ''' Actually evaluate the objective function '''
        self.results = np.zeros(self.npoints)
        for s,sample in enumerate(self.samples): # TODO: parallelize
            self.results[s] = self.func(sample, **self.func_args) # This is the time-consuming step!!
        self.allresults[self.key] = sc.dcp(self.results)
        return self.results
    
    
    def step(self):
        ''' Calculate new samples based on the current samples and matching results '''
    
        # Calculate ordinary least squares fit of sample parameters on results
        if self.optimum == 'max': results =  self.results # Default, just use the stored results
        else:                     results = -self.results # Flip the sign if we're using the minimum
        mod = sm.OLS(results, sm.add_constant(self.samples))
        mod_fit = mod.fit() # Perform fit
        
        # Decide what type of step to take
        fitinds = sc.findinds(self.fittable)
        xranges = self.xmax - self.xmin
        old_center = self.x[fitinds]
        if mod_fit.rsquared > self.mp.rsquared_thresh: # The hyperplane is a good fit, calculate gradient descent
            coef = mod_fit.params[1:]  # Drop constant
            den = np.sqrt(sum([xranges[p]**2 * c**2 for c,p in zip(coef, fitinds)]))
            scale = self.relstepsize*max(self.mp.mu_r, self.mp.sigma_r)
            new_center = [xi + xranges[p]**2 * c*scale/den for xi, c, p in zip(old_center, coef, fitinds)]
            if self.mp.useadaptation:
                self.relstepsize *= self.mp.adaptation['step']
                self.relstepsize = np.median([self.mp.adaptation['min'], self.relstepsize, self.mp.adaptation['max']]) # Set limits
        else: # It's a bad fit, just pick the best point
            max_idx = np.argmax(results)
            new_center = self.samples[max_idx]
            if self.mp.useadaptation:
                self.relstepsize *= self.mp.adaptation['step']**(np.random.choice([-1,1]))
                correction = 1.89 # Corrective factor so mean(log(abs(correction*randn()))) ≈ 0
                dist = np.linalg.norm((new_center - old_center)/xranges) # Normalized distance to the best point
                if self.mp.mu_r: # Shell-based sampling
                    self.relstepsize = dist/self.mp.mu_r # Get the ratio of the new distance and the current distance
                else:
                    self.relstepsize = correction*dist/self.mp.sigma_r
                self.relstepsize = np.median([self.mp.adaptation['min'], self.relstepsize, self.mp.adaptation['max']]) # Set limits
            
        
        # Update values
        self.x[fitinds] = new_center # Reassign center
        self.x = np.minimum(self.xmax, np.maximum(self.xmin, self.x)) # Clamp
        self.allcenters[self.key] = sc.dcp(self.x)
        self.sample_hypershell() # Calculate new hypershell and return
        print(self.relstepsize)
        return self.samples
    
    
    def optimize(self):
        ''' Actually perform an optimization '''
        self.sample_hypershell() # Initialize
        for i in range(self.maxiters): # Iterate
            if self.verbose>=1: print(f'Step {i+1} of {self.maxiters}')
            self.evaluate() # Evaluate the objective function
            self.step() # Calculate the next step
        
        # Create output structure
        output = sc.objdict()
        output['x'] = self.x # Parameters
        output['fval'] = self.func(self.x, **self.func_args) # TODO: consider placing this elsewhere
        output['exitreason'] = f'Reached maximum iteration limit ({self.maxiters})' # Stopping conditions not really implemented yet
        output['obj'] = self # Just return the entire original object
        return output
            


def dwits(*args, **kwargs):
    '''
    Wrapper for DWITS class
    '''
    dwest = DWITS(*args, **kwargs) # Create class instance
    output = dwest.optimize() # Run the optimization
    return output
    
    
    
    
    
    
    
    
