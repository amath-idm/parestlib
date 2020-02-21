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