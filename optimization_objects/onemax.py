# all values at max
import mlrose_hiive as mlrose
import numpy as np


def get_one_max():
    one_max = mlrose.OneMax()
    state = np.array([0,1,0,1,1,1,1])
    one_max.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=10,
        fitness_fn=one_max,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
