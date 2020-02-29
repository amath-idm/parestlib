'''
Export defaults from dtk-tools to run standalone EMOD.

WARNING, the exported files don't work! You need to run convert18to20
and then also manually fix each of the errors (of which there are about 
half a dozen) before it will actually run.
'''


from dtk.utils.core.DTKConfigBuilder import DTKConfigBuilder
from simtools.SetupParser import SetupParser

SetupParser.init()
cb = DTKConfigBuilder.from_defaults('MALARIA_SIM')
cb.dump_files('/u/cliffk/idm/parestlib/emod_examples/malaria/input')

print('Done')