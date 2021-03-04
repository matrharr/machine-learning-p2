import mlrose_hiive as mlrose
import numpy as np

def get_six_peaks():
    six_peaks = mlrose.SixPeaks(t_pct=0.1)
    state = np.array([0,1,0,1,1,1,1])
    six_peaks.evaluate(state)
    problem = mlrose.DiscreteOpt(
        length=10,
        fitness_fn=six_peaks,
        maximize=True,
        max_val=2 # makes it bit string
    )
    return problem
