import time
import mlrose_hiive as mlrose
import numpy as np

from algos.hill import get_hill_climb
from optimization_objects.continuous_peaks import get_continuous_peaks
from optimization_objects.flip_flop import get_flip_flop
from optimization_objects.four_peaks import get_four_peaks
from optimization_objects.knapsack import get_knapsack
from optimization_objects.onemax import get_one_max
from optimization_objects.six_peaks import get_six_peaks
from visualization.plot_graphs import plot_hill_restarts

input_sizes = [
    25, 50, 75, 100, 150, 200, 300
]

opt_probs = [
    (get_one_max, 'one max'),
    (get_four_peaks, 'four peaks'),
    (get_six_peaks, 'six peaks'),
    (get_continuous_peaks, 'continuous peaks'),
    (get_flip_flop, 'flip flop'),
]

hill = get_hill_climb

options = [
    0, 10, 20, 40
]

metric_dict = { prob[1]: {} for prob in opt_probs }

for size in input_sizes:
    print('####################################### New input size ', size,' ##############################################')
    for prob in opt_probs:
        problem_name = prob[1]
        problem = prob[0](size)
        print('----------------------------------New Problem ', problem_name, ' ------------------------------------------')
        for opt in options:
            print('___________New Option_____________')
            print('Optimization Problem: ', problem_name)
            print('Input size: ', size)
            print('Option: ', opt)

            start = time.time()
            best_state, best_fitness, fitness_curve = hill(problem, opt)
            end = time.time()
            time_taken = end - start
            print('execution time taken: ', time_taken)
            
            print('best state: ', best_state)
            print('best fitness: ', best_fitness)
            print('number of iterations: ', len(fitness_curve))

            if metric_dict[problem_name].get(opt) is not None:
                metric_dict[problem_name][opt]['iterations'].append(len(fitness_curve))
                metric_dict[problem_name][opt]['time_taken'].append(time_taken)
            else:
                metric_dict[problem_name][opt] = { 
                    'iterations': [len(fitness_curve)],
                    'time_taken': [time_taken]
                }

metric_dict['input_sizes'] = input_sizes
print(metric_dict)
print(metric_dict['one max'][10]['iterations'])

plot_hill_restarts(metric_dict)