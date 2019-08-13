import optim_methods as om

om.plot_problem(which='norm', ndims=2, noise=0.3)
om.plot_problem(which='norm', ndims=3, noise=0.3)
om.plot_problem(which='rosenbrock', ndims=2, noise=0.3)
om.plot_problem(which='rosenbrock', ndims=3, noise=0.3)

print('Done.')