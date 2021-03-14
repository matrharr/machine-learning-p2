import mlrose_hiive as mlrose
import numpy as np 

def get_mimic(problem, keep_pct=0.25):
    
    best_state, best_fitness, fitness_curve = mlrose.mimic(
        problem,
        pop_size=200,
        keep_pct=keep_pct,
        max_attempts=10,
        max_iters=np.inf,
        curve=True,
        random_state=23,
    )

    return best_state, best_fitness, fitness_curve