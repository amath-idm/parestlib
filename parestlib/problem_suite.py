'''
A suite of test problems for optimization algorithms.

See the tests folder for example usages.

Version: 2019aug14
'''


##########################################
### Housekeeping
##########################################


import pylab as pl
import sciris as sc

__all__ = ['addnoise', 'make_norm', 'make_rosenbrock', 'make_hills', 'plot_problem']


def addnoise(err, noise=0.0):
    '''
    Add noise to the objective function. If noise is a scalar, then use it as a value
    and supply defaults. If noise is a dict, get kwargs from it.

    Examples:
        # Add multiplicative Gaussian noise with std=0.1 to an objective value of 3.67
        addnoise(3.67, noise=0.1)

        # Add additive uniform noise with range=0.1 to an objective value of 3.67
        addnoise(3.67, noise={'value':0.1, 
                              'gaussian':False,
                              'nonnegative':True,
                              'multiplicative':False,
                              'verbose':True})
    '''

    # Ensure it's a dict and set properties
    if not isinstance(noise, dict): noise = {'value':noise}
    value          = noise.get('value', 0.0)
    gaussian       = noise.get('gaussian', True)
    nonnegative    = noise.get('nonnegative', True)
    multiplicative = noise.get('multiplicative', True)
    verbose        = noise.get('verbose', True)

    if value: # If no noise, just skip to the end

        # Create the zero-mean noise
        if gaussian: noiseterm = value*pl.randn()
        else:        noiseterm = value*(pl.rand()-0.5)

        # Update the error
        if multiplicative: output = err * (1 + noiseterm)
        else:              output = err + noiseterm

        # If asked, swap the sign of the output -- WARNING, will mess up the statistical properties!
        if nonnegative and pl.sign(output) != pl.sign(err):
            if verbose: print('addnoise() warning: noise reversed sign of error; reversing back')
            output = -output

    else:
        output = err

    return output



##########################################
### Define test problems
##########################################

def make_norm(noise=0.0, optimum=None, verbose=True, delay=None):
    '''
    Simplest problem possible -- just the norm of the input vector.
    '''
    if optimum is None: optimum = 'min'

    def norm(pars, noise=0.0, optimum='min', delay=None):
        if delay:
            pl.pause(delay*(0.5+0.5*pl.rand())) # Add a noticeable delay
        err = pl.linalg.norm(pars)
        err = addnoise(err, noise)
        if optimum == 'max':
            err = -err
        return err

    func = lambda pars: norm(pars, noise=noise, optimum=optimum, delay=delay) # Create the actual function

    if verbose:
        print("Created test norm function with noise=%s" % (noise))
        print('Suggested starting point: [1, 1, ... 1]')
        print('Suggested limits: [-1,1]')
        print('Optimal solution: %s=0 at [0, 0, ... 0]' % optimum)
    return func


def make_rosenbrock(ndims=2, noise=0.0, optimum=None, verbose=True):
    '''
    Make a Rosenbrock's valley of 2 or 3 dimensions, optionally with noise.
    '''
    if optimum is None: optimum = 'min'

    def rosenbrock(pars, ndims=2, noise=0.0, optimum='min'):
        x = pars[0]
        y = pars[1]
        err = 50*(y - x**2)**2 + (0.5 - x)**2; # Rosenbrock's valley
        if ndims == 3: # Optionally add a 3rd dimension
            z = pars[2]
            err += 10*abs(z-0.5)
        elif ndims > 3:
            raise NotImplementedError
        err = addnoise(err, noise)
        if optimum == 'max':
            err = -err
        return err

    func = lambda pars: rosenbrock(pars, ndims=ndims, noise=noise, optimum=optimum) # Create the actual function

    if verbose:
        print("Created test Rosenbrock's valley function with ndims=%s, noise=%s" % (ndims, noise))
        print('Suggested starting point: %s' % ([-1]*ndims))
        print('Suggested limits: [-1,1]')
        print('Optimal solution: %s=0 at %s' % (optimum, [0.5]*ndims))
    return func



def make_hills(noise=0.0, optimum=None, verbose=True):
    '''
    Test problem from:
        Marchant R, Ramos F. 
        Bayesian optimisation for Intelligent Environmental Monitoring. 
        In: IEEE International Conference on Intelligent Robots and Systems 2012:2242-2249
        doi:10.1109/IROS.2012.6385653
    '''
    if optimum is None: optimum = 'max'

    def hills(pars, noise=0.0, optimum='max'):
        x1 = pars[0]
        x2 = pars[1]
        err = pl.exp(-(x1-4)**2)*pl.exp(-(x2-1)**2) + 0.8*pl.exp(-(x1-1)**2)*pl.exp(-((x2-4)/2.5)**2) + 4*pl.exp(-((x1-10)/5)**2)*pl.exp(-((x2-10)/5)**2)
        err = addnoise(err, noise)
        if optimum == 'min':
            err = -err
        return err

    func = lambda pars: hills(pars, noise=noise, optimum=optimum) # Create the actual function

    if verbose:
        print("Created test hills function with noise=%s" % (noise))
        print('Suggested starting point: [0.5,0.5]')
        print('Suggested limits: [0,5]')
        print('Optimal solution: %s≈1.037 near [4,1]' % optimum)
    return func


####################################################
### Plot test problems -- see tests/plot_problems.py
####################################################

def plot_problem(which='rosenbrock', ndims=3, noise=None, npts=None, startvals=None, 
                 minvals=None, maxvals=None, randseed=None, perturb=None, alpha=None, 
                 uselog=True, force3d=False, trajectory=None, optimum=None, verbose=False):
    ''' Plot one of the test problems '''

    if ndims !=2 and which not in ['norm', 'rosenbrock']:
        if verbose: print('Note: ndims=%s not supported for %s, resetting to 2' % (ndims, which))
        ndims = 2

    if startvals is None: startvals = -1*pl.ones(ndims)
    if minvals   is None: minvals   = -1*pl.ones(ndims)
    if maxvals   is None: maxvals   =  1*pl.ones(ndims)
    if ndims == 2: # Set defaults
        if noise    is None: noise = 0.0
        if npts     is None: npts  = 100
        if perturb  is None: perturb = 0.0
        if alpha    is None: 
            if force3d: alpha = 0.5
            else:       alpha = 1.0
    elif ndims == 3:
        if noise    is None: noise = 0.0
        if npts     is None: npts  = 15
        if perturb  is None: perturb = 0.05
        if alpha    is None: alpha = 0.5
    else:
        raise NotImplementedError

    # Make vectors
    if randseed:
        pl.seed(randseed)
    xvec = pl.linspace(minvals[0],maxvals[0],npts)
    yvec = pl.linspace(minvals[1],maxvals[1],npts)
    if ndims == 2: zvec = [0]
    else:          zvec = pl.linspace(minvals[2],maxvals[2],npts)

    # Define objective function
    if   which == 'rosenbrock': objective_func = make_rosenbrock(ndims=ndims, noise=noise, optimum=optimum)
    elif which == 'norm':       objective_func = make_norm(noise=noise, optimum=optimum)
    elif which == 'hills':      objective_func = make_hills(noise=noise, optimum=optimum)
    else:                       objective_func = which # Assume it's supplied directly

    # Evaluate at each point
    alldata = []
    for x in xvec:
        for y in yvec:
            for z in zvec:
                xp = x + perturb*pl.randn()
                yp = y + perturb*pl.randn() 
                zp = z + perturb*pl.randn() 
                o = objective_func([xp, yp, zp])
                alldata.append([xp, yp, zp, o])
    alldata = pl.array(alldata)
    X = alldata[:,0]
    Y = alldata[:,1]
    Z = alldata[:,2]
    O = alldata[:,3]
    if uselog:
        if (O>=0).all(): # Calculate the log -- normal case
            O = pl.log10(O) 
        elif (O<=0).all(): # Completely negative -- flip, calculate the log, then flip again
            O = -O # Flip the sign
            O = pl.log10(O) # Calculate the log
            O = -O # Flip back
        else:
            print('WARNING: plot_problem() cannot plot the log since the values cross 0')
    fig = pl.figure(figsize=(16,12))
    if ndims == 2:
        if force3d:
            ax = sc.scatter3d(X, Y, O, O, fig=fig, plotkwargs={'alpha':alpha})
            ax.view_init(elev=90, azim=0)
        else:
            ax = pl.scatter(X, Y, c=O, alpha=alpha)
            pl.colorbar()
    else:
        ax = sc.scatter3d(X, Y, Z, O, fig=fig, plotkwargs={'alpha':alpha})
        ax.view_init(elev=50, azim=-45)
    pl.xlabel('x')
    pl.ylabel('y')

    # Plot trajectory
    if trajectory:
        X2 = trajectory[:,0]
        Y2 = trajectory[:,1]
        if ndims == 2:
            O2 = pl.log10(trajectory[:,2])
            ax = sc.scatter(X2, Y2, c=O2, marker='d')
            ax = sc.plot3d(X2, Y2, c=(0,0,0), lw=3)
        else:
            Z2 = trajectory[:,2]
            O2 = pl.log10(trajectory[:,3])
            ax = sc.scatter3d(X2, Y2, Z2, O2, fig=fig, plotkwargs={'alpha':1.0, 'marker':'d'})
            ax = sc.plot3d(X2, Y2, Z2, fig=fig, plotkwargs={'c':(0,0,0), 'lw':3})

    return fig


