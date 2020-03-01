'''
Run tests on each of the methods.

Optimizations run are problems * methods * repeats; comment out lines that aren't
required.

Version: 2020mar01
'''

import sciris as sc
import parestlib as pe

problems = [
        'norm',
        'rosenbrock',
        ]

startvals = {
        'norm': [1, 2, 3],
        'rosenbrock': [-1, -1, -1]
        }

methods = {
        'ASD':       pe.asd,
        'BSD':       pe.bsd,
        'ShellStep': pe.shellstep,
        'DWITS':     pe.dwits,
        }

repeats = 3
noisevals = [0, 0.05] # For noise values of larger than 0.05, standard ASD breaks
#if 'doplot' not in locals(): doplot = True # For future use if plotting is implemented


results = []
for method,optim_func in methods.items():
    for problem in problems:
        sc.heading(f'Running {method}() on {problem}')
        for n,noise in enumerate(noisevals):
            
            # Define the problem
            if   problem == 'norm':       objective_func = pe.make_norm(noise=noise, verbose=0)
            elif problem == 'rosenbrock': objective_func = pe.make_rosenbrock(ndims=len(startvals), noise=noise, verbose=0)
            else:                         raise NotImplementedError
            
            for r in range(repeats):
                print(f'\nRun {n*repeats+r+1} of {repeats*len(noisevals)} with noise={noise}:')
                result = optim_func(objective_func, startvals[problem], verbose=0)
                results.append(result)
                print(f'  Iterations: {len(result.details.fvals)}\n  Value: {result.fval}\n  Result: {result.x}')