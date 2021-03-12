# total number of pairs of consecutive elements

import mlrose_hiive as mlrose
import numpy as np

def get_flip_flop(size):
    flip_flop = mlrose.FlipFlop()
    state = np.array([0,1,0,1,1,1,1])
    flip_flop.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=size,
        fitness_fn=flip_flop,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
