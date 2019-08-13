'''
Reimplementation of Dan K.'s OptimTool from dtk-tools.

Version: 2019aug13
'''

import numpy as np
import sciris as sc

__all__ = ['optimtool']

def optimtool(func, pars, verbose=2):
    output = sc.objdict()
    output['x'] = [] # Parameters
    output['fval'] = np.nan
    output['exitreason'] = 'n/a'
    output['details'] = sc.objdict()
    output['details']['fvals'] = [] # Function evaluations
    output['details']['xvals'] = []
    return output
