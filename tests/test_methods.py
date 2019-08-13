'''
Run tests on ASD.

Version: 2019aug13
'''

import sciris as sc
import optim_methods as om

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
        'OptimTool': om.optimtool
        }

repeats = 3
noisevals = [0, 0.05] # For noise values of larger than 0.05, standard ASD breaks

#if 'doplot' not in locals(): doplot = True # For future use if plotting is implemented

def heading(string):
    sc.colorize('blue', '\n'*3+'—'*10+' '+string+' '+'—'*10)
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
                result = optim_func(objective_func, startvals[problem], verbose=0)
                results.append(result)
                print('  Iterations: %s\n  Value: %s\n  Result: %s' % (len(result.details.fvals), result.fval, result.x))