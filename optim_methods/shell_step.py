'''
Reimplementation of Dan K.'s OptimTool from dtk-tools.

The naming convention is to distinguish the module (shell_step) from the function
(shellstep).

Basic usage is:
    
    import optim_methods as om
    output = om.shellstep(func, x, xmin, xmax)

Version: 2019aug18
'''

import numpy as np
import sciris as sc
import scipy.special as sp
import scipy.stats as st
import statsmodels.api as sm

__all__ = ['get_r', 'ShellStep', 'shellstep']

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
        Radius of the hypershell
    
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


class ShellStep(sc.prettyobj):
    '''
    Reimplementation of the original dtk-tools OptimTool function:
        dtk-tools/calibtool/algorithms/OptimTool.py
    
    The algorithm has two basic parts. In the first part, it samples from a hypershell
    around the current point. In the second part, it takes this hypershell and tries to fit
    a hyperplane to it, and if it "succeeds" (the r-squared value is high enough),
    it will step in the uphill direction. Otherwise, it just takes the best point.
    
    Version: 2019aug18
    '''
    
    def __init__(self, func, x, xmin, xmax, fittable=None, mp=None, maxiters=None, optimum=None, func_args=None, verbose=None):
        self.func = func
        self.x    = np.array(x)
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.fittable  = np.array(fittable) if fittable is not None else np.ones(len(x)) # Set everything to be fittable by default
        self.maxiters  = maxiters  if maxiters  is not None else 10 # Set iterations to be 10 by default
        self.func_args = func_args if func_args is not None else {}
        self.verbose   = verbose   if verbose   is not None else 2
        self.optimum   = optimum   if optimum   is not None else 'max'
        self.set_mp(mp) # mp = metaparameters; can be None, 'sphere', 'shell', or a dict of values
        self.samples = None
        self.results = None
        self.iteration = 0
        self.relstepsize = 1.0 # The current relative step size
        self.allcenters = sc.odict({self.key: self.x}) # Clunky way of creating a dict with a string key the iteration
        self.allsamples = sc.odict() # Initialize storing the value of x on each iteration
        self.allresults = sc.odict() # Initialize storing the results for each iteration
        return
    
    
    def set_mp(self, mp=None):
        '''
        Calculate default metaparameters, and override with user-supplied values.
        With no arguments, returns the default metaparameters.
        '''
        npars = sum(self.fittable)
        vfrac = 0.01
        r = get_r(npars, vfrac)
        self.mp = sc.objdict({
                    'mu_r':    0, # By default, use a sphere rather than a shell
                    'sigma_r': r,
                    'N':       10,
                    'center_repeats': 1,
                    'rsquared_thresh': 0.5,
                    'adaptation': {
                            'step': np.sqrt(2),
                            'min': 0.1,
                            'max': 10}
                    })
        if mp == 'sphere' or mp is None: # By default, use a sphere of spread sigma_r = r
            self.mp.mu_r = 0
            self.mp.sigma_r = r # Use in place of mu_r
        elif mp == 'shell': # Optionally, use a shell of radius mu_r = r and spread sigma_r = r/10
            self.mp.mu_r = r
            self.mp.sigma_r = r/10.
        else: # Assume it's a dict and update
            self.mp.update(mp)
        
        if self.mp.mu_r == 0 and self.mp.sigma_r == 0:
            raise Exception('Either mu_r or sigma_r must be greater than 0')
        
        self.mp = sc.objdict(self.mp) # Ensure it's an objdict for dot access (e.g. self.mp.mu_r)
        return self.mp
    
    
    @property
    def key(self):
        ''' Use the current iteration number, as a string, to set the key for the dicts '''
        return str(self.iteration)
        
    
    def sample_hypershell(self):
        ''' Sample points from a hypershell. '''
        
        # Initialize
        fitinds = sc.findinds(self.fittable) # The indices of the fittable parameters
        nfittable = len(fitinds) # How many fittable parameters
        npars = len(self.x) # Total number of parameters
        standard_normal = st.norm(loc=0, scale=1) # Initialize a standard normal distribution
        radius_normal = st.norm(loc=self.relstepsize*self.mp.mu_r, scale=self.relstepsize*self.mp.sigma_r) # And a scaled one
        
        # Calculate deviations
        deviations = np.zeros((self.mp.N, nfittable)) # Deviations from current center point
        for r in range(self.mp.N): # Loop over repeats
            sn_rvs = standard_normal.rvs(size=nfittable) # Sample from the standard distribution
            sn_nrm = np.linalg.norm(sn_rvs) # Calculate the norm of these samples
            radius = radius_normal.rvs() # Sample from the scaled distribution
            deviations[r,:] = radius/sn_nrm*sn_rvs # Deviation is the scaled sample adjusted by the rescaled standard sample
        
        # Calculate parameter samples
        samples = np.zeros((self.mp.N, npars)) # New samples
        for p in range(npars): # Loop over all parameters
            if self.fittable[p]:
                ind = sc.findinds(fitinds==p)[0] # Convert the parameter index back to the fittable parameter index
                delta = deviations[:,ind] * (self.xmax[p] - self.xmin[p]) # Scale the deviation by the allowed parameter range
            else:
                delta = 0 # If not fittable, set to zer
            samples[:,p] = self.x[p] + delta # Set new parameter value
        
        # Overwrite with center repeats
        for r in range(self.mp.center_repeats):
            samples[r,:] = self.x # Just replace with the current center
        
        # Clamp
        for p in range(npars):
            samples[:,p] = np.minimum(self.xmax[p], np.maximum(self.xmin[p], samples[:,p])) # Ensure all samples are within range
        
        self.samples = samples
        self.allsamples[self.key] = sc.dcp(samples)
        self.iteration += 1
        return self.samples
    
    
    def evaluate(self):
        ''' Actually evaluate the objective function '''
        self.results = np.zeros(self.mp.N)
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
            if self.mp.adaptation:
                self.relstepsize *= self.mp.adaptation['step']
                self.relstepsize = np.median([self.mp.adaptation['min'], self.relstepsize, self.mp.adaptation['max']]) # Set limits
        else: # It's a bad fit, just pick the best point
            max_idx = np.argmax(results)
            new_center = self.samples[max_idx]
            if self.mp.adaptation:
                self.relstepsize *= self.mp.adaptation['step']**(np.random.choice([-1,1]))
#                dist = np.linalg.norm((new_center - old_center)/xranges) # Normalized distance to the best point
#                if self.mp.mu_r: # Shell-based sampling
#                    self.relstepsize = dist/self.mp.mu_r # Get the ratio of the new distance and the current distance
#                else:
#                    self.relstepsize = dist/self.mp.sigma_r
                self.relstepsize = np.median([self.mp.adaptation['min'], self.relstepsize, self.mp.adaptation['max']]) # Set limits
            
        
        # Update values
        self.x[fitinds] = new_center # Reassign center
        self.x = np.minimum(self.xmax, np.maximum(self.xmin, self.x)) # Clamp
        self.allcenters[self.key] = sc.dcp(self.x)
        self.sample_hypershell() # Calculate new hypershell and return
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
            


def shellstep(*args, **kwargs):
    '''
    Wrapper for ShellStep class
    '''
    shest = ShellStep(*args, **kwargs) # Create class instance
    output = shest.optimize() # Run the optimization
    return output
    
    
    
    
    
    
    
    
