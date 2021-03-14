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
from visualization.plot_graphs import plot_iterations_vs_probsize

input_sizes = [
    25, 50, 75, 100, 300
]

opt_probs = [
    (get_one_max, 'one max'),
    (get_four_peaks, 'four peaks'),
    (get_six_peaks, 'six peaks'),
    (get_continuous_peaks, 'continuous peaks'),
    (get_flip_flop, 'flip flop'),
    # (get_knapsack, 'knapsack') # requires auto scale weight vector
]

algo_dict = {
    'hill_climb': get_hill_climb,
    'genetic': get_genetic,
    'mimic': get_mimic,
    'sim_ann': get_sim_ann
}

metric_dict = { prob[1]: {} for prob in opt_probs }

for size in input_sizes:
    print('####################################### New input size ', size,' ##############################################')
    for prob in opt_probs:
        problem_name = prob[1]
        problem = prob[0](size)
        print('----------------------------------New Problem ', problem_name, ' ------------------------------------------')
        for algo in algo_dict.keys():
            print('___________New Algorithm_____________')
            print('Optimization Problem: ', problem_name)
            print('Algorithm: ', algo)
            print('Input size: ', size)

            start = time.time()
            best_state, best_fitness, fitness_curve = algo_dict[algo](problem)
            end = time.time()
            time_taken = end - start
            print('execution time taken: ', time_taken)
            
            print('best state: ', best_state)
            print('best fitness: ', best_fitness)
            print('number of iterations: ', len(fitness_curve))

            if metric_dict[problem_name].get(algo) is not None:
                metric_dict[problem_name][algo]['iterations'].append(len(fitness_curve))
                metric_dict[problem_name][algo]['time_taken'].append(time_taken)
            else:
                metric_dict[problem_name][algo] = { 
                    'iterations': [len(fitness_curve)],
                    'time_taken': [time_taken]
                }

metric_dict['input_sizes'] = input_sizes
print(metric_dict)
# print(metric_dict['one max']['hill_climb']['iterations'])
# metric_dict = {
#     'one max': {
#         'hill_climb': {
#             'iterations': [169, 218, 396, 598, 739, 1420], 
#             'time_taken': [0.1047980785369873, 0.13292717933654785, 0.18542098999023438, 0.3240189552307129]}, 
#         'genetic': {
#             'iterations': [107, 122, 126, 132, 156, 174], 
#             'time_taken': [2.1138980388641357, 2.3254168033599854, 2.6096479892730713, 4.505553960800171]}, 
#         'mimic': {
#             'iterations': [14, 17, 19, 21, 28, 42], 
#             'time_taken': [3.3200740814208984, 15.437178134918213, 38.77648591995239, 143.82218408584595]}, 
#         'sim_ann': {
#             'iterations': [403, 397, 399, 491, 973, 1114], 
#             'time_taken': [0.013805150985717773, 0.009961366653442383, 0.010024785995483398, 0.021817922592163086]}}, 
#     'continuous peaks': {
#         'hill_climb': {
#             'iterations': [235, 568, 448, 453, 385, 363], 
#             'time_taken': [0.15097594261169434, 0.43291592597961426, 0.7756400108337402, 0.5779132843017578]}, 
#         'genetic': {
#             'iterations': [162, 266, 431, 269, 418, 607], 
#             'time_taken': [3.243450880050659, 8.740301847457886, 21.552452087402344, 10.277268171310425]}, 
#         'mimic': {
#             'iterations': [18, 20, 22, 25, 27, 25], 
#             'time_taken': [4.154268026351929, 24.659236192703247, 60.7720148563385, 106.88955307006836]}, 
#         'sim_ann': {
#             'iterations': [360, 927, 3668, 3932, 12012, 42572],
#             'time_taken': [0.013003110885620117, 0.2650871276855469, 0.2511920928955078, 0.3071470260620117]}}, 
#     'flip flop': {
#         'hill_climb': {
#             'iterations': [162, 267, 214, 315, 423, 797], 
#             'time_taken': [0.2736320495605469, 1.2937629222869873, 1.5953700542449951, 2.3168017864227295]}, 
#         'genetic': {
#             'iterations': [135, 155, 169, 185, 233, 252], 
#             'time_taken': [4.261296987533569, 9.16793704032898, 16.58466410636902, 20.93815779685974]}, 
#         'mimic': {
#             'iterations': [15, 15, 21, 23, 39, 31], 
#             'time_taken': [3.5401461124420166, 14.28897476196289, 68.44921612739563, 137.29828596115112]},
#         'sim_ann': {
#             'iterations': [940, 680, 2852, 2643, 17867, 18519], 
#             'time_taken': [0.07252192497253418, 0.09671974182128906, 0.8462467193603516, 1.362076997756958]}},
#     'input_sizes': [25, 50, 75, 100, 200, 300]}
plot_iterations_vs_probsize(metric_dict)

# {
#     'one max': {
#         'hill_climb': {
#             'iterations': []
#         }
#     }
# }
