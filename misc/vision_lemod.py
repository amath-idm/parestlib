
#%% User interface

import pylab as pl
import lemod_fp as lfp

pars = sp.make_pars()
data = 'UN_demographic_data.csv'
sim = lfp.Sim(pars=pars, data=data)
sim.calibrate()


#%% Internal implementation

import parestlib as om

def calibrate(self, n_iters=100):

    def objective_fn(x):
        results = self.run(pars=x)
        mismatch = 0
        for i in len(self.data):
            mismatch += ((results[i] - self.data[i])**2).sum()
        return mismatch

    pars = om.shellstep(objective_fn, x=x, xmin=xmin, xmax=xmax)
    self.pars_from_vec(pars)

    return None
