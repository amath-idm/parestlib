'''
Run tests on each of the methods.

Optimizations run are problems * methods * repeats; comment out lines that aren't
required.

Version: 2019aug13
'''

import sciris as sc
import parestlib as om
import numpy as np

problems = [
    'norm',
    'rosenbrock',
]

startvals = {
    'norm': [1, 2, 3],
    'rosenbrock': [-1, -1, -1]
}

methods = {
    'ASD':       om.asd,
    #'BSD':       om.bsd,
    'ShellStep': om.shellstep,
}

repeats = 3
noisevals = [0, 0.05] # For noise values of larger than 0.05, standard ASD breaks
#if 'doplot' not in locals(): doplot = True # For future use if plotting is implemented

def heading(string):
    divider = '—'*15 # Unicode, watch out!
    sc.colorize('blue', '\n'*3+divider+string+divider)
    return None

results = []
for method,optim_func in methods.items():
    for problem in problems:
        heading('Running %s on %s()' % (method, problem))
        for n,noise in enumerate(noisevals):

            # Define the problem
            if   problem == 'norm':       objective_func = om.make_norm(noise=noise, verbose=0)
            elif problem == 'rosenbrock': objective_func = om.make_rosenbrock(ndims=len(startvals), noise=noise, verbose=0)
            else:                         raise NotImplementedError

            for r in range(repeats):
                print('\nRun %s of %s with noise=%s:' % (n*repeats+r+1, repeats*len(noisevals), noise))
                kwargs = {}
                if method == 'ShellStep':
                    kwargs = {
                        'xmin' : [-5, -5, -5],
                        'xmax' : [5, 5, 5], # TODO: Error if guess is outside of xmin / xmax
                        'maxiters': 20,
                        'mp' : {
                            'mu_r': 0.05,
                            'sigma_r': 0.01,
                            'N': 20,
                            'center_repeats': 1,
                            'rsquared_thresh': 0.5,
                            'useadaptation': True,
                            'adaptation': {
                                    'step': 0.9, # Should be less than 1
                                    'min': 0.05, # More than 0
                                    'max': 1.4
                            }
                        },
                    }
                result = optim_func(objective_func, startvals[problem], optimum='min', verbose=0, **kwargs)
                results.append(result)
                # DJK N.B. Iterations is different from the number of funcation evaluations!
                # In supercomputing, we also aim to reduce iterations
                iters = np.shape(np.array(result.details.fvals))[0]
                fevals = np.size(np.array(result.details.fvals))
                print(  f'\tIterations: {iters}\n',
                        f'\tFunction Evaluations: {fevals}',
                        f'\tValue: {result.fval}',
                        f'\tResult: {result.x}')
