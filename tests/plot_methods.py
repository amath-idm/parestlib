'''
Plot the stochastic descent algorithm.

Version: 2019aug18
'''

import pylab as pl
import optim_methods as om

# Choose the problem and method
problem = ['norm', 'rosenbrock', 'hills'][1]
method = ['shellstep', 'asd'][0]
maxiters = 50
doplot = 1
uselog = 1
geometry = 3
randseed = 7159 #20935 # None # 37854

# Set noise level
pl.seed(randseed)
usenoise = 1
if usenoise:
    noise = {'value':0.5, # Amount of noise to add -- defaults 0.3, 1, 1, 0
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
elif problem == 'hills':
    initial = [0.5, 0.5]
    minvals = [0.0, 0.0]
    maxvals = [5.0, 5.0]

# Create the problem
if   problem == 'norm':        objective_func = om.make_norm(noise=noise, optimum='min')
elif problem == 'rosenbrock':  objective_func = om.make_rosenbrock(noise=noise, optimum='min')
elif problem == 'hills':       objective_func = om.make_hills(noise=noise, optimum='min')

# Plot the objective function
if doplot:
    om.plot_problem(which=problem, ndims=2, noise=noise, optimum='min', uselog=uselog, minvals=minvals, maxvals=maxvals)


# Perform the optimization
if method == 'shellstep':
    output = om.shellstep(func=objective_func, 
                          x=initial, 
                          xmin=minvals, 
                          xmax=maxvals,
                          optimum='min',
                          mp=['original',  # 0 -- geometry option
                              'shell',     # 1
                              'sphere',    # 2
                              {'mu_r':0.2, # 3
                               'sigma_r':0.02, 
                               'N':50, 
                               'useadaptation':1}][geometry],
                          maxiters=maxiters)
    samples = output.obj.allsamples # Make this easier 
elif method == 'asd':
    output = om.asd(function=objective_func, 
                    x=initial, 
                    xmin=minvals, 
                    xmax=maxvals,
                    optimum='min',
                    maxiters=maxiters,
                    verbose=2)
    samples = output.details.xvals # Make this easier 


# Animate the results
if doplot:
    delay = 0.1
    ax = pl.gca()
    dots = None
    for i in range(len(samples)):
        if dots is not None: dots.remove()
        if method == 'shellstep':
            dots = pl.scatter(samples[i][:,0], samples[i][:,1], c=[(1,0.9,1)], marker='s')
        elif method == 'asd':
            dots = pl.scatter(samples[i][0], samples[i][1], c=[(1,0.9,1)], marker='s')
        ax.set_title(f'Iteration: {i+1}/{len(samples)}')
        pl.xlim((minvals[0], maxvals[0]))
        pl.ylim((minvals[1], maxvals[1]))
        pl.pause(delay)

print(f'Final parameters: {output.x}')
print(f'Final result: {output.fval}')

print('Done.')