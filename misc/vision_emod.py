#%% User interface

import pylab as pl
import emodpy as ep

# Set the algorithm metaparameters
algorithm = 'shellstep' # Which optimization algorithm to use
how = 'parallel' # How to run (local, parallel, or COMPS)
n_iters = 10 # Maximum iterations of calibration algorithm
n_samples = 20 # Number of samples per iteration

# Define the objective of the calibration
def objective_fn(sim):
    orig_infected = pl.array(orig_results.data['Channels']['Infected']['Data'])
    this_infected = pl.array(sim.results.data['Channels']['Infected']['Data'])
    mismatch = ((orig_infected - this_infected)**2).sum()
    return mismatch

# Choose which parameters to calibrate
parameters = [{'name': 'Base_Infectivity',
              'best': 0.0001,
              'min': 0.0,
              'max': 0.1}]

# Run calibration
sim = ep.Simulation()
calib = ep.Calibration(sim=sim, parameters=parameters, objective_fn=objective_fn, algorithm=algorithm)
results = calib.run(how=how, n_iters=n_iters)



#%% Internal implementation

import parestlib as om
from . import em_simulations as ems
from . import em_experiments as ees

class Calibration:
    
    def run(self, objective_fn=None, algorithm=None, ...):
        
        # Set algorithm
        if algorithm == 'shellstep':
            optim_func = om.shellstep
        elif algorithm in ['asd', 'ASD']:
            optim_func = sc.asd
        else:
            raise NotImplementedError
        
        # Define default run function
        def run_fn(x):
            sim = ems.Simulation(**sim_kwargs)
            for v,val in enumerate(x):
                par_name = parameters[v]['name']
                sim.config['parameters'][par_name] = val
            exp = ees.Experiment(sims=[sim], **exp_kwargs)
            exp.run(**run_kwargs)
            sim.results = exp.results.results[0] # Warning, make simpler!
            error = objective_fn(sim)
            return error
        
        output = optim_func(run_fn, x=x, xmin=xmin, xmax=xmax, **optim_kwargs)
        return output