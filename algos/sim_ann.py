import mlrose_hiive as mlrose
import numpy as np 

def get_sim_ann(problem):
    
    best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
        problem,
        schedule=mlrose.GeomDecay(),
        max_attempts=10,
        max_iters=np.inf,
        init_state=None,
        curve=True,
        random_state=23
    )

    return best_state, best_fitness, fitness_curve