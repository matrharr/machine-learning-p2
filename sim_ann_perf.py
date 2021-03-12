import time
import mlrose_hiive as mlrose
import numpy as np

from algos.sim_ann import get_sim_ann
from optimization_objects.continuous_peaks import get_continuous_peaks
from optimization_objects.flip_flop import get_flip_flop
from optimization_objects.four_peaks import get_four_peaks
from optimization_objects.knapsack import get_knapsack
from optimization_objects.onemax import get_one_max
from optimization_objects.six_peaks import get_six_peaks
from visualization.plot_graphs import plot_sim_ann_schedules

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

sim_ann = get_sim_ann

options = [
    (mlrose.GeomDecay(), 'geometric'), 
    (mlrose.ArithDecay(), 'arithmetic'),
    (mlrose.ExpDecay(), 'exponential')
]

metric_dict = { prob[1]: {} for prob in opt_probs }

for size in input_sizes:
    print('####################################### New input size ', size,' ##############################################')
    for prob in opt_probs:
        problem_name = prob[1]
        problem = prob[0](size)
        print('----------------------------------New Problem ', problem_name, ' ------------------------------------------')
        for opt in options:
            option_name = opt[1]
            opt_val = opt[0]
            print('___________New Option_____________')
            print('Optimization Problem: ', problem_name)
            print('Input size: ', size)
            print('Option: ', option_name)

            start = time.time()
            best_state, best_fitness, fitness_curve = sim_ann(problem, opt_val)
            end = time.time()
            time_taken = end - start
            print('execution time taken: ', time_taken)
            
            print('best state: ', best_state)
            print('best fitness: ', best_fitness)
            print('number of iterations: ', len(fitness_curve))

            if metric_dict[problem_name].get(option_name) is not None:
                metric_dict[problem_name][option_name]['iterations'].append(len(fitness_curve))
                metric_dict[problem_name][option_name]['time_taken'].append(time_taken)
            else:
                metric_dict[problem_name][option_name] = { 
                    'iterations': [len(fitness_curve)],
                    'time_taken': [time_taken]
                }

metric_dict['input_sizes'] = input_sizes
print(metric_dict)
print(metric_dict['one max']['geometric']['iterations'])

plot_sim_ann_schedules(metric_dict)