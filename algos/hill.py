import mlrose_hiive as mlrose
import numpy as np 

def get_hill_climb(problem, restarts=0):
    
    best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(
        problem,
        max_attempts=100,
        max_iters=np.inf,
        restarts=restarts,
        init_state=None,
        curve=True,
        random_state=23
    )

    return best_state, best_fitness, fitness_curve