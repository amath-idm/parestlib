import pandas as pd
import pylab as pl
import parestlib as pe
from history_matching.gpr import GPR
from history_matching.basis import Basis


# Set input data parameters
ntrain = 200
ntest  = 50
npars  = 2
noise  = 0.3

def gen_vals(raw, noise, dist='uniform'):
    ''' Calculate the noisy distance from the center '''
    output = pl.sqrt(((raw-raw.mean())**2).sum(axis=1))
    if dist == 'uniform':
        output += noise*(pl.rand(len(output)))
    elif dist == 'gaussian':
        output += noise*(pl.randn(len(output)))
    return output

# Set up training and test arrays
train_arr = pl.rand(ntrain, npars)
train_vals = gen_vals(train_arr, noise=noise)
test_arr = pl.rand(ntest, npars)
test_vals = gen_vals(test_arr, noise=noise)

par_names = [f'x{i}' for i in range(npars)]
param_info = pd.DataFrame( {'Min':[0,0], 'Max':[1,1]}, index=par_names)

# Prepare the training data
Ycol = 'y'
train_data = pd.DataFrame(train_arr, columns=par_names)
train_data[Ycol] = train_vals
train_data.index.name = 'Sample_Id'

# Prepare the test data
test_data = pd.DataFrame(test_arr, columns=par_names)
test_data.index.name = 'Sample_Id'


# Define the basis - should just default to all in param_info!
basis = Basis.identity_basis(param_info.index, param_info=param_info)

# Create an instance of the GPR
gpr = GPR(basis, Ycol, train_data, param_info, log_transform=False)

# Fit the GPR
gpr.optimize_hyperparameters(
    x0 = [0.5, 0.5] + [0.1]*npars,
    bounds = ((0.05,100), (0.1,10)) + ((0.01,1),)*npars
)

# Calculate the estimates
ret = gpr.evaluate(test_data)

test_vals_knn = pe.bootknn(test=test_arr, train=train_arr, values=train_vals) 

# Plot results
fig, ax = pl.subplots(1,2,figsize=(16,10))

if npars >= 2:
    ax[0].scatter(train_arr[:,0], train_arr[:,1], c=train_vals, marker='o')
    ax[0].scatter(test_arr[:,0], test_arr[:,1], c=test_vals_knn.best, marker='s', s=50)
    ax[0].scatter(test_arr[:,0], test_arr[:,1], c=ret['Mean'], marker='o')
    ax[0].set_xlabel(par_names[0])
    ax[0].set_ylabel(par_names[1])

ax[1].plot([min(test_vals),max(test_vals)], [min(test_vals),max(test_vals)], 'r-')
knn_err_low  = test_vals_knn.best - test_vals_knn.low
knn_err_high = test_vals_knn.high - test_vals_knn.best
gp_err = 2*pl.sqrt(ret['Var_Latent'])
ax[1].errorbar(x=test_vals, y=test_vals_knn.best, yerr=[knn_err_low, knn_err_high], fmt='o', c='b', lw=0.5, label='KNN')
ax[1].errorbar(x=test_vals, y=ret['Mean'], yerr=gp_err, fmt='o', c='g', lw=1, label='GP')
ax[1].legend()
ax[1].set_xlabel('True Output')
ax[1].set_ylabel('Predicted Output')

pl.show()
