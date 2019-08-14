import optim_methods as om

kwargs = dict(
        uselog  = 0,   # Whether or not to use a logarithmic scale -- default 1
        noise   = 0.0, # Amount of noise to add -- default 0.3
        force3d = 1    # Whether to show 2D plots in 3D -- default 0
        )



om.plot_problem(which='norm', ndims=2, **kwargs)
#om.plot_problem(which='norm', ndims=3, n**kwargs)
om.plot_problem(which='rosenbrock', ndims=2, **kwargs)
#om.plot_problem(which='rosenbrock', ndims=3, **kwargs)
#om.plot_problem(which='hills', ndims=2, minvals=[0,0], maxvals=[5,5], **kwargs)

print('Done.')