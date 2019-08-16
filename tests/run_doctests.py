
import doctest
import optim_methods as om

doctest.testmod(om.optim_tool, verbose=True)

#import os
#import sciris as sc
#testdir = os.path.join(os.pardir, 'optim_methods')
#filenames = sc.getfilelist(folder=testdir, ext='py')
#for filename in filenames:
#    doctest.testfile(filename, verbose=True)