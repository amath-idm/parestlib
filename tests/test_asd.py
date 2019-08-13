'''
Run tests on ASD
'''

import sciris as sc
import optim_methods as om

torun = [
'norm',
'rosenbrock',
]

repeats = 3

if 'doplot' not in locals(): doplot = True

def heading(string):
    sc.colorize('blue', '\n'*3+'—'*10+' '+string+' '+'—'*10)
    return None

if 'norm' in torun:
    # For noise values of larger than 0.05, standard ASD breaks
    heading('Running ASD on norm()')
    noisevals = [0, 0.05]
    startvals = [1, 2, 3]
    norm_results = []
    for n,noise in enumerate(noisevals):
        for r in range(repeats):
            print('\nRun %s of %s with noise=%s:' % (n*repeats+r+1, repeats*len(noisevals), noise))
            func = om.make_norm(noise=noise, verbose=0)
            result = sc.asd(func, startvals, verbose=0)
            norm_results.append(result)
            print('  Iterations: %s\n  Value: %s\n  Result: %s' % (len(result.details.fvals), result.fval, result.x))


if 'rosenbrock' in torun:
    # For noise values of larger than 0.05, standard ASD breaks
    heading('Running ASD on rosenbrock()')
    noisevals = [0, 0.05]
    startvals = [-1, -1, -1]
    rosenbrock_results = []
    for n,noise in enumerate(noisevals):
        for r in range(repeats):
            print('\nRun %s of %s with noise=%s:' % (n*repeats+r+1, repeats*len(noisevals), noise))
            func = om.make_rosenbrock(ndims=len(startvals), noise=noise, verbose=0)
            result = sc.asd(func, startvals, verbose=0)
            rosenbrock_results.append(result)
            print('  Iterations: %s\n  Value: %s\n  Result: %s' % (len(result.details.fvals), result.fval, result.x))
