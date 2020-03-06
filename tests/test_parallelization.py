'''
Run parallelization tests

Version: 2020mar04
'''

#%% Imports

import sciris as sc
import parestlib as pe


#%% Initialization

kwargs = {
    'optimum': 'min',
    'verbose': 0,
    'xmin' : [-5, -5, -5],
    'xmax' : [5, 5, 5], # TODO: Error if guess is outside of xmin / xmax
    'maxiters': 10,
    'mp' : {
        'mu_r': 0.05,
        'sigma_r': 0.01,
        'N': 10,
        'center_repeats': 1,
        'rsquared_thresh': 0.5,
        'useadaptation': True,
        'adaptation': {
                'step': 0.9, # Should be less than 1
                'min': 0.05, # More than 0
                'max': 1.4
        }
    }
}

startvals = [1, 2, 3]

lambda_func = pe.make_norm(noise=0.05, verbose=0, delay=0.05) # pe.make_rosenbrock(ndims=len(startvals), noise=0.05, verbose=0)

def objective_func(*args, **kwargs):
    ''' We have to wrap the lambda function since can't be passed to multiprocessing directly '''
    return lambda_func(*args, **kwargs)

#%% The tests

def test_parallelization(doplot=False):
    sc.heading('Testing parallelization...')
    
    print('Running in serial')
    t1i = sc.tic()
    serial_result = pe.shellstep(objective_func, startvals, parallelize=False, **kwargs)
    t1f = sc.toc(t1i, output=True)
    
    print('Running in parallel')
    t2i = sc.tic()
    parallel_result = pe.shellstep(objective_func, startvals, parallelize=True, **kwargs)
    t2f = sc.toc(t2i, output=True)
    
    print(f'Serial time: {t1f:0.3f} s; parallel time: {t2f:0.3f} s')
    return serial_result, parallel_result


#%% Run as a script -- comment out lines to turn off tests
if __name__ == '__main__':
    sc.tic()
    result = test_parallelization()
    print('\n'*2)
    sc.toc()
    print('Done.')