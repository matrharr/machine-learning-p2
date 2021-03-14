import mlrose_hiive as mlrose
import numpy as np 

def get_sim_ann(problem, schedule=mlrose.GeomDecay()):
    
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=100,
        max_iters=np.inf,
        init_state=None,
        curve=True,
        random_state=2
    )

    return best_state, best_fitness, fitness_curve