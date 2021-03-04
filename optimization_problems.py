'''
imports optimization objects
imports algos
runs each algo on each optimization object
'''
import time
import mlrose_hiive as mlrose
import numpy as np

from algos.genetic import get_genetic
from algos.hill import get_hill_climb
from algos.mimic import get_mimic
from algos.sim_ann import get_sim_ann
from optimization_objects.continuous_peaks import get_continuous_peaks
from optimization_objects.flip_flop import get_flip_flop
from optimization_objects.four_peaks import get_four_peaks
from optimization_objects.knapsack import get_knapsack
from optimization_objects.onemax import get_one_max
from optimization_objects.six_peaks import get_six_peaks

one_max = get_one_max()
four_peaks = get_four_peaks()
six_peaks = get_six_peaks()
continuous_peaks = get_continuous_peaks()
flip_flop = get_flip_flop()
knapsack = get_knapsack()

opt_probs = [
    (one_max, 'one max'),
    (four_peaks, 'four peaks'),
    (six_peaks, 'six peaks'),
    (continuous_peaks, 'continuous peaks'),
    (flip_flop, 'flip flop'),
    (knapsack, 'knapsack')
]

algo_dict = {
    'hill_climb': get_hill_climb,
    'genetic': get_genetic,
    'mimic': get_mimic,
    'sim_ann': get_sim_ann
}

for prob in opt_probs:
    print('----------------------------------New Problem ', prob[1], ' ------------------------------------------')
    for algo in algo_dict.keys():
        print('___________New Algorithm_____________')
        print('Optimization Problem: ', prob[1])
        print('Algorithm: ', algo)

        start = time.time()
        best_state, best_fitness, fitness_curve = algo_dict[algo](prob[0])
        end = time.time()
        print('execution time taken: ', end - start)
        
        print('best state: ', best_state)
        print('best fitness: ', best_fitness)

        
{
    'one max': {
        'hill_climb': {
            'time': 
            'score': 
        }
    }
}