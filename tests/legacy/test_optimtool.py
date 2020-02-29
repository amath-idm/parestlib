'''
Tests for OptimTool. Adapted from dtk-tools/examples/Calibration/example_OptimTool.py

NOTE -- this file is deprecated, it was used to test equivalence between the new
and original versions of OptimTool (now called ShellStep).

Version: 2019aug18
'''

##########################################
### Housekeeping
##########################################

import pylab as pl
import pandas as pd
import sciris as sc
from calibtool.algorithms.OptimTool import OptimTool
import parestlib as om

# Set parameters
randseed = 5845235
torun = [
#        'initial_points',
        'iterate',
        ]

# To make dataframes easier to debug
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)



def create_DTK():
    ''' Create an OptimTool instance '''
    
    # Overwrite default repr with something more informative
    def new_repr(self): return sc.prepr(self)
    
    
    def gather_results(self, randseed=None, results=None):
        if randseed is not None:
            pl.seed(randseed)
        if results is None:
            results = pl.rand(len(self.data['Results']))
        self.data['Results'][-len(results):] = results
        return results
    
    OptimTool.__repr__ = new_repr
    OptimTool.gather_results = gather_results

    params = [
        {
            'Name': 'Clinical Fever',
            'Dynamic': True,
            #'MapTo': 'Clinical_Fever_Threshold_High', # <-- DEMO: Custom mapping, see map_sample_to_model_input below
            'Guess': 1.75,
            'Min': 0.5,
            'Max': 2.5
        },
        {
            'Name': 'MSP1 Merozoite',
            'Dynamic': False,   # <-- NOTE: this parameter is frozen at Guess
            'MapTo': 'MSP1_Merozoite_Kill_Fraction',
            'Guess': 0.65,
            'Min': 0.4,
            'Max': 0.7
        },
        {
            'Name': 'Falciparum',
            'Dynamic': True,
            'MapTo': 'Falciparum_PfEMP1_Variants',
            'Guess': 1500,
            'Min': 1,  # 900 [0]
            'Max': 5000 # 1700 [1e5]
        },
        {
            'Name': 'Min Days',
            'Dynamic': False,  # <-- NOTE: this parameter is frozen at Guess
            'MapTo': 'Min_Days_Between_Clinical_Incidents',
            'Guess': 25,
            'Min': 1,
            'Max': 50
        },
    ]
    
    num_params = len([p for p in params if p['Dynamic']])
    
    volume_fraction = 0.01   # desired fraction of N-sphere area to unit cube area for numerical derivative (automatic radius scaling with N)
    r = OptimTool.get_r(num_params, volume_fraction)
    OT = OptimTool(params,
        mu_r=r,             # <-- radius for numerical derivatve.  CAREFUL not to go too small with integer parameters
        sigma_r=r/10.,      # <-- stdev of radius
        center_repeats=0,   # <-- Number of times to replicate the center (current guess).  Nice to compare intrinsic to extrinsic noise
        samples_per_iteration=10  # 32 # <-- Samples per iteration, includes center repeats.  Actual number of sims run is this number times number of sites.
    )
    
    return OT


class create_OM(sc.prettyobj):
    ''' Wrapper class for parestlib functions, for easier testing '''
    
    def __init__(self):
        self.x        = pl.array([1.75, 0.65, 1500, 25])
        self.xmin     = pl.array([0.50, 0.40,    1,  1])
        self.xmax     = pl.array([2.50, 0.70, 5000, 50])
        self.fittable = pl.array([   1,    0,    1,  0])
        npars = sum(self.fittable)
        vfrac = 0.01
        r = om.optim_tool.get_r(npars, vfrac)
        self.mp = sc.objdict({
                    'mu_r':    r,
                    'sigma_r': r/10,
                    'N':       10,
                    'center_repeats': 1,
                    'rsquared_thresh': 0.5,
                    })
        return
    
    def sample_hypersphere(self):
        samples = om.optim_tool.sample_hypersphere(mp=self.mp, x=self.x, xmin=self.xmin, xmax=self.xmax, fittable=self.fittable)
        return samples
    
    def gather_results(self, randseed=None, results=None):
        if randseed is not None:
            pl.seed(randseed)
        if results is None:
            results = pl.rand(self.mp.N)
        return results
    
    def ascend(self, samples, results):
        samples = om.optim_tool.ascend(samples, results, mp=self.mp, x=self.x, xmin=self.xmin, xmax=self.xmax, fittable=self.fittable)
        return samples
    
    

# Create the class instances to be used for the tests
OT = create_DTK()
OM = create_OM()


##########################################
### Run tests
##########################################

if 'initial_points' in torun:
    # Tests to run
    doprint = False
    doplot = False
    doassert = True
    
    # Choose samples
    pl.seed(randseed)
    dtk_samples_df = OT.choose_initial_samples()
    pl.seed(randseed)
    om_samples = OM.sample_hypersphere()
    dtk_samples = dtk_samples_df.to_numpy()
    
    # Tests
    if doprint:
        print(dtk_samples)
        print(om_samples)
    
    if doplot:
        fig = pl.figure()
        pl.subplot(2,1,1)
        pl.hist(dtk_samples[:,0], bins=100)
        pl.subplot(2,1,2)
        pl.hist(om_samples[:,0], bins=100)
    
    if doassert:
        try:
            print('Asserting equality of initial points...')
            assert (om_samples==dtk_samples).all()
            print('Passed!')
        except:
            raise
            
    
if 'iterate' in torun:
    # Tests to run
    doprint = True
    doassert = True
    doplot = True
    niters = 10
    
    def results_from_sample(sample):
        sample = pl.array(sample) # Ensure it's an array
        variates = sample[:,[0,2]]
        norms = pl.norm(variates, axis=1)
        return norms
        
    # DTK
    pl.seed(randseed)
    dtk_sam = OT.choose_initial_samples()
    dtk_sams = [dtk_sam]
    dtk_results = []
    for i in range(niters):
        sams = dtk_sams[-1]
        results = results_from_sample(sams)
        OT.gather_results(results=results)
        new_sams = OT.choose_samples_via_gradient_ascent(i+1)
        dtk_sams.append(new_sams)
        dtk_results.append(results)
    
    # parestlib
    pl.seed(randseed)
    om_sam = OM.sample_hypersphere() 
    om_sams = [om_sam]
    om_results = []
    for i in range(niters):
        sams = om_sams[-1]
        results = results_from_sample(sams)
        OM.gather_results(results=results)
        new_sams = OM.ascend(sams, results)
        om_sams.append(new_sams)
        om_results.append(results)
    
    if doprint:
        print(dtk_sams[-1])
        print(om_sams[-1])
    
    if doassert:
        try:
            print('Asserting equality (within rounding error) of the final set of sample points...')
            assert sc.approx(om_sams[-1], dtk_sams[-1].to_numpy(), eps=1e-6).all()
            print('Passed!')
        except:
            raise
    
    
    

print('Done.')
