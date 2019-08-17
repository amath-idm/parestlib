'''
Tests for OptimTool. Adapted from dtk-tools/examples/Calibration/example_OptimTool.py

Version: 2019aug17
'''

##########################################
### Housekeeping
##########################################

import pylab as pl
import sciris as sc
from calibtool.algorithms.OptimTool import OptimTool
import optim_methods as om

# Set parameters
randseed = 5845235
torun = [
        'initial_points',
        ]



def create_DTK():
    ''' Create an OptimTool instance '''

    params = [
        {
            'Name': 'Clinical Fever Threshold High',
            'Dynamic': True,
            #'MapTo': 'Clinical_Fever_Threshold_High', # <-- DEMO: Custom mapping, see map_sample_to_model_input below
            'Guess': 1.75,
            'Min': 0.5,
            'Max': 2.5
        },
        {
            'Name': 'MSP1 Merozoite Kill Fraction',
            'Dynamic': False,   # <-- NOTE: this parameter is frozen at Guess
            'MapTo': 'MSP1_Merozoite_Kill_Fraction',
            'Guess': 0.65,
            'Min': 0.4,
            'Max': 0.7
        },
        {
            'Name': 'Falciparum PfEMP1 Variants',
            'Dynamic': True,
            'MapTo': 'Falciparum_PfEMP1_Variants',
            'Guess': 1500,
            'Min': 1,  # 900 [0]
            'Max': 5000 # 1700 [1e5]
        },
        {
            'Name': 'Min Days Between Clinical Incidents',
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
        samples_per_iteration=10000  # 32 # <-- Samples per iteration, includes center repeats.  Actual number of sims run is this number times number of sites.
    )
    return OT


class create_OM():
    ''' Wrapper class for optim_methods functions, for easier testing '''
    
    def __init__(self):
        self.x        = pl.array([1.75, 0.65, 1500, 25])
        self.xmin     = pl.array([0.50, 0.40,    1,  1])
        self.xmax     = pl.array([2.50, 0.70, 5000, 50])
        self.fittable = pl.array([   1,    0,    1,  0])
        npars = len(self.x)
        vfrac = 0.01
        r = om.optim_tool.get_r(npars, vfrac)
        self.mp = sc.objdict({
                    'mu_r':    r,
                    'sigma_r': r/10,
                    'N':       10000,
                    'center_repeats': 1,
                    })
        return None
    
    def sample_hypersphere(self):
        samples = om.optim_tool.sample_hypersphere(mp=self.mp, x=self.x, xmin=self.xmin, xmax=self.xmax, fittable=self.fittable)
        return samples
    
    


OT = create_DTK()
OM = create_OM()


##########################################
### Run tests
##########################################

if 'initial_points' in torun:
    doprint = False
    doplot = True
    dtk_samples_df = OT.choose_initial_samples()
    om_samples = OM.sample_hypersphere()
    dtk_samples = dtk_samples_df.to_numpy()
    
    if doprint:
        print(dtk_samples)
        print(om_samples)
    
    if doplot:
        fig = pl.figure()
        pl.subplot(2,1,1)
        pl.hist(dtk_samples[:,0], bins=100)
        pl.subplot(2,1,2)
        pl.hist(om_samples[:,0], bins=100)
    
    

print('Done.')
