'''
Plot the test problems in problem_suite.py.
'''

import pylab as pl
import sciris as sc
import optim_methods as om

kwargs = dict(
        uselog  = 1,   # Whether or not to use a logarithmic scale -- default 1
        noise   = 0,   # Not implemented since function is already noisy
        force3d = 1    # Whether to show 2D plots in 3D -- default 0
        )

# Plot the blowfly simulation
fig1 = pl.figure(figsize=(16,12))
output1 = om.plot_blowflies(fig=fig1)
output2 = om.plot_blowflies(fig=fig1)
output3 = om.plot_blowflies(fig=fig1)
pl.legend(('Sim 1', 'Sim 2', 'Sim 3'))
pl.pause(0.3) # Has to be long enough for the figure to fully render

# Plot the error function
sc.tic()
blowflies = om.make_blowflies()
print('Simulating blowflies (takes 30-60 s)...')
om.plot_problem(which=blowflies, minvals=[0,0], maxvals=[80,2], **kwargs)
pl.xlabel('r')
pl.ylabel('σ')
sc.toc()