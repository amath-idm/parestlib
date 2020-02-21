class Calibration:
    
    def run(self, objective_fn=None, algorithm=None, ...):
        
        # Set algorithm
        if algorithm == 'shellstep':
            optim_func = om.shellstep
        elif algorithm in ['asd', 'ASD']:
            optim_func = sc.asd
        else:
            raise NotImplementedError
        
        # Define default run function
        def run_fn(x):
            sim = ems.Simulation(**sim_kwargs)
            for v,val in enumerate(x):
                par_name = parameters[v]['name']
                sim.config['parameters'][par_name] = val
            exp = ees.Experiment(sims=[sim], **exp_kwargs)
            exp.run(**run_kwargs)
            sim.results = exp.results.results[0] # Warning, make simpler!
            error = objective_fn(sim)
            return error
        
        output = optim_func(run_fn, x=x, xmin=xmin, xmax=xmax, **optim_kwargs)
        return output