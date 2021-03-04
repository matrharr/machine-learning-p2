import mlrose_hiive as mlrose
import numpy as np

def get_continuous_peaks():
    continuous_peaks = mlrose.ContinuousPeaks(t_pct=0.1)
    state = np.array([0,1,0,1,1,1,1])
    continuous_peaks.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=10,
        fitness_fn=continuous_peaks,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
