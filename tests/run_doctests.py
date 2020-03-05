'''
Run the doctests on the imported modules. If you want to run on files instead,
use this syntax:

    import os
    import sciris as sc
    testdir = os.path.join(os.pardir, 'parestlib')
    filenames = sc.getfilelist(folder=testdir, ext='py')
    for filename in filenames:
        doctest.testfile(filename, verbose=True)

Version: 2019aug16
'''

import doctest
import parestlib as om
import sciris as sc

# See __init__.py
submodules = [
        'problem_suite',
        'sim_suite',
        'optim_tool',
        'stochastic_descent',
        ]

results = {}
for name in submodules:
    submodule = getattr(om, name)
    sc.heading(f'Running tests on: {name}')
    result = doctest.testmod(submodule, verbose=True)
    results[name] = result


sc.heading('Summary:')
attempted = 0
failed = 0
for key,val in results.items():
    attempted += val.attempted
    failed += val.failed
    print(f'  %20s: attempted={val.attempted}, failed={val.failed}' % key)

sc.heading('Bottom line:')
if failed:
    print(f'WARNING: {failed} tests failed!')
else:
    print(f'All {attempted} tests passed!')

