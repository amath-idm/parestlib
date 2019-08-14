import optim_methods as om

kwargs = dict(
        uselog  = 1,   # Whether or not to use a logarithmic scale -- default 1
        noise   = {'value':0.3, # Amount of noise to add -- default 0.3, 1, 1, 0
                   'gaussian':1, 
                   'multiplicative':1,
                   'verbose':0}, 
        force3d = 0    # Whether to show 2D plots in 3D -- default 0
        )

om.plot_problem(which='norm', ndims=2, **kwargs)
om.plot_problem(which='norm', ndims=3, **kwargs)
om.plot_problem(which='rosenbrock', ndims=2, **kwargs)
om.plot_problem(which='rosenbrock', ndims=3, **kwargs)
om.plot_problem(which='hills', ndims=2, minvals=[0,0], maxvals=[5,5], **kwargs)

print('Done.')