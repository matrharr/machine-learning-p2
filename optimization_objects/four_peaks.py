# good for genetic algos

import mlrose_hiive as mlrose
import numpy as np

def get_four_peaks():
    four_peaks = mlrose.FourPeaks(t_pct=0.1)
    state = np.array([0,1,0,1,1,1,1])
    four_peaks.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=10,
        fitness_fn=four_peaks,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
