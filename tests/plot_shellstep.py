'''
Plot the shellstep algorithm.

Version: 2019aug18
'''

import optim_methods as om

problem = 'norm'

om.plot_problem(which='norm', ndims=2, optimum='max', uselog=False)

print('Done.')