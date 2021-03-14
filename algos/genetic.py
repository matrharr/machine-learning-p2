import mlrose_hiive as mlrose
import numpy as np 

def get_genetic(problem, pop_size=300, mutation_prob=0.4):
    
    best_state, best_fitness, fitness_curve = mlrose.genetic_alg(
        problem,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
        max_attempts=100,
        max_iters=np.inf,
        curve=True,
        random_state=23
    )

    return best_state, best_fitness, fitness_curve
