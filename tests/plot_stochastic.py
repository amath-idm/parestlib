'''
Plot the stochastic descent algorithm.

Version: 2019aug18
'''

import pylab as pl
import optim_methods as om

problem = 'rosenbrock'

# Set noise level
usenoise = 0
if usenoise:
    noise = {'value':0.2, # Amount of noise to add -- defaults 0.3, 1, 1, 0
             'gaussian':1, 
             'multiplicative':1,
             'verbose':0}
else:
    noise = None

# Set the initial, minimum, and maximum values
if problem == 'norm':
    initial = [ 100.,   50.]
    minvals = [-100., -100.]
    maxvals = [ 100.,  100.]
elif problem == 'rosenbrock':
    initial = [-1., -1.]
    minvals = [-1., -1.]
    maxvals = [ 1.,  1.]

# Create the problem
if problem == 'norm':
    objective_func = om.make_norm(noise=noise, optimum='min')
elif problem == 'rosenbrock':
    objective_func = om.make_rosenbrock(noise=noise, optimum='min')

# Perform the optimization
output = om.stochastic_descent.asd(function=objective_func, 
                      x=initial, 
                      xmin=minvals, 
                      xmax=maxvals,
                      optimum='min',
                      maxiters=50,
                      verbose=2)
samples = output.details.xvals # Make this easier 

# Plot the objective function
om.plot_problem(which=problem, ndims=2, noise=noise, optimum='min', uselog=False, minvals=minvals, maxvals=maxvals)

# Animate the results
delay = 0.2
ax = pl.gca()
dots = None
for i in range(len(samples)):
    if dots is not None: dots.remove()
    dots = pl.scatter(samples[i][0], samples[i][1], c=[[0.8]*3])
    ax.set_title(f'Iteration: {i+1}/{len(samples)}')
    pl.pause(delay)

print(f'Final parameters: {output.x}')
print(f'Final result: {output.fval}')

print('Done.')