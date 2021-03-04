import mlrose_hiive as mlrose
import numpy as np 

def get_genetic(problem):
    
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
        problem,
        pop_size=200,
        mutation_prob=0.1,
        max_attempts=10,
        max_iters=np.inf,
        curve=True,
        random_state=23
    )

    return best_state, best_fitness, fitness_curve