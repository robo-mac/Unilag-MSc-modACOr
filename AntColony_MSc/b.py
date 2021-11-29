import time
startTime = time.time()

import numpy as np

from a import ACOm
#def __init__(self, distance, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
'''distance = np.array([[np.inf, 2, 2, 5, 7],
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 5],
                      [5, 8, 1, np.inf, 2],
                      [7, 2, 7, 2, np.inf]])'''

"""distance = np.random.rand(5,5)"""
distance = np.array([[0.0090, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132],
					[0.0140, 0.0150, 0.0162, 0.0173, 0.0180, 0.0200],
					[0.0230, 0.0250, 0.0280, 0.0320, 0.0350, 0.0410],
					[0.0470, 0.0540, 0.0630, 0.0720, 0.0800, 0.0920],
					[0.1050, 0.1200, 0.1350, 0.1480, 0.1620, 0.1770],
					[0.1920, 0.2070, 0.2250, 0.2440, 0.2630, 0.2830]])
"""distance = np.array([4, 6],
                    [9, 4])"""

"""x = np.inf
distance = np.array([[np.inf, 2, 2, 5, 7],
                    [2, np.inf, 4, 8, 2],
                    [2, 4, np.inf, 1, 5],
                    [5, 8, 1, np.inf, 2],
                    [7, 2, 7, 2, np.inf]])"""

#Hartman
'''distance = np.array([4.0, 4.0, 4.0, 4.0],
                    [1.0, 1.0, 1.0, 1.0],
                    [8.0, 8.0, 8.0, 8.0],
                    [6.0, 6.0, 6.0, 6.0],
                    [3.0, 7.0, 3.0, 7.0],
                    [2.0, 9.0, 2.0, 9.0],
                    [5.0, 5.0, 3.0, 3.0],
                    [8.0, 1.0, 8.0, 1.0],
                    [6.0, 2.0, 6.0, 2.0],
                    [7.0, 3.6, 7.0, 3.6])'''

a = ACOm(distance, 2, 1, 1000, 0.85, alpha=0, beta=0)
shortest_path = a.run()
#pheromone_cost = a.spread_pheronome()
executionTime = (time.time() - startTime)
print ("shortest_path: {}".format(shortest_path))
print('Execution time in seconds is: '+ str(executionTime))
"""include Eval_Func_Count: Number of times the objective function is called
	Experiment: = Number of Sets of runs we will implement
	Iterations: = Number of times we run the search in each experiment
	min_best: ExperimentBest
	avg_best: AverageBest
	Expand_Number"""