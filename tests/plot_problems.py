'''
Plot the test problems in problem_suite.py.
'''

import pylab as pl
import parestlib as pe

kwargs = dict(
        uselog  = 1,   # Whether or not to use a logarithmic scale -- default 1
        noise   = {'value':0.3, # Amount of noise to add -- default 0.3, 1, 1, 0
                   'gaussian':1, 
                   'multiplicative':1,
                   'verbose':0}, 
        force3d = 1    # Whether to show 2D plots in 3D -- default 0
        )

pe.plot_problem(which='norm', ndims=2, **kwargs)
pe.plot_problem(which='norm', ndims=3, **kwargs)
pe.plot_problem(which='rosenbrock', ndims=2, **kwargs)
pe.plot_problem(which='rosenbrock', ndims=3, **kwargs)
pe.plot_problem(which='hills', ndims=2, minvals=[0,0], maxvals=[5,5], **kwargs)
pl.show()


print('Done.')