'''
Plot the shellstep algorithm.

Version: 2019aug18
'''

import pylab as pl
import optim_methods as om

# Set noise level
usenoise = False
if usenoise:
    noise = {'value':0.3, # Amount of noise to add -- defaults 0.3, 1, 1, 0
             'gaussian':1, 
             'multiplicative':1,
             'verbose':0}
else:
    noise = None


# Perform the optimization
objective_func = om.make_rosenbrock(noise=noise, optimum='min')
output = om.shellstep(func=objective_func, 
                      x=[1,1], 
                      xmin=[-1,-1], 
                      xmax=[1,1], 
                      optimum='min')
samples = output.obj.allsamples # Make this easier 

# Plot the objective function
om.plot_problem(which='rosenbrock', ndims=2, noise=noise, optimum='min', uselog=False)

# Animate the results
delay = 1.0
ax = pl.gca()
dots = None
for i in range(len(samples)):
    if dots is not None: dots.remove()
    dots = pl.scatter(samples[i][:,0], samples[i][:,1], c=(1,1,1))
    ax.set_title(f'Iteration: {i+1}/{len(samples)}')
    pl.pause(delay)


print('Done.')