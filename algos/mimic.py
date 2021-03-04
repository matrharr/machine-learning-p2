import mlrose_hiive as mlrose
import numpy as np 

def get_mimic(problem):
    
    best_state, best_fitness, fitness_curve = mlrose.mimic(
        problem,
        pop_size=200,
        keep_pct=0.2,
        max_attempts=10,
        max_iters=np.inf,
        curve=True,
        random_state=23,
        # fast_mimic=False
    )

    return best_state, best_fitness, fitness_curve