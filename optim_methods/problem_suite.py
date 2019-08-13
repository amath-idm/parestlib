'''
A suite of test problems for optimization algorithms.

Version: 2019aug13
'''

import pylab as pl
import sciris as sc

__all__ = ['make_norm', 'make_rosenbrock', 'plot_problem']


def addnoise(err, noise=0.0):
    if noise:
        noiseterm = 1 + noise*pl.randn()
        if noiseterm<0: noiseterm = -noiseterm # Don't allow it to go negative
        err *= noiseterm
    return err


def make_norm(noise=0.0):
    '''
    Simplest problem possible -- just the norm of the input vector.
    '''
    def norm(pars, noise=0.0):
        err = pl.linalg.norm(pars)
        err = addnoise(err, noise)
        return err
    
    func = lambda pars: norm(pars, noise=noise)
    print("Created test norm function with noise=%s" % (noise))
    print('Optimal solution: [0, 0, ... 0]')
    return func


def make_rosenbrock(ndims=2, noise=0.0):
    '''
    Make a Rosenbrock's valley of 2 or 3 dimensions, optionally with noise.
    '''
    
    def rosenbrock(pars, ndims=2, noise=0.0):
        x = pars[0]
        y = pars[1]
        err = 50*(y - x**2)**2 + (0.5 - x)**2; # Rosenbrock's valley
        if ndims == 3:
            z = pars[2]
            err += 10*abs(z-0.5)
        elif ndims > 3:
            raise NotImplementedError
        err = addnoise(err, noise)
        return err

    func = lambda pars: rosenbrock(pars, ndims=ndims, noise=noise)
    print("Created test Rosenbrock's valley function with ndims=%s, noise=%s" % (ndims, noise))
    print('Suggested starting point: %s' % ([-1]*ndims))
    print('Optimal solution: %s' % ([0.5]*ndims))
    return func
    

def plot_problem(which='rosenbrock', ndims=3, noise=None, npts=None, startvals=None, minvals=None, maxvals=None, randseed=None, perturb=None, alpha=None, trajectory=None):
    if startvals is None: startvals = -1*pl.ones(ndims)
    if minvals   is None: minvals   = -1*pl.ones(ndims)
    if maxvals   is None: maxvals   =  1*pl.ones(ndims)
    if ndims == 2:
        if noise    is None: noise = 0.0
        if npts     is None: npts  = 100
        if perturb  is None: perturb = 0.0
        if alpha    is None: alpha = 1.0
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
    if which == 'rosenbrock':
        func = make_rosenbrock(ndims=ndims, noise=noise)
    elif which == 'norm':
        func = make_norm(noise=noise)
    else:
        raise NotImplementedError
    
    # Evaluate at each point
    alldata = []
    for x in xvec:
        for y in yvec:
            for z in zvec:
                xp = x + perturb*pl.randn()
                yp = y + perturb*pl.randn() 
                zp = z + perturb*pl.randn() 
                objective = func([xp, yp, zp])
                o = pl.log10(objective)
                alldata.append([xp, yp, zp, o])
    alldata = pl.array(alldata)
    X = alldata[:,0]
    Y = alldata[:,1]
    Z = alldata[:,2]
    O = alldata[:,3]
    fig = pl.figure(figsize=(16,12))
    if ndims == 2:
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


