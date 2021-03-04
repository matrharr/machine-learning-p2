import mlrose_hiive as mlrose
import numpy as np

def get_knapsack():
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    max_weight_pct = 0.6
    knapsack = mlrose.Knapsack(weights, values, max_weight_pct)
    state = np.array([1, 0, 2, 1, 0])
    knapsack.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=5,
        fitness_fn=knapsack,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
