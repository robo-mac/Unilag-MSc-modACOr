'''
    ==============================================================
    Ant Colony Optimization algorithm for continuous domains ACO_R
    ==============================================================

    author: GreenBrush
    Github:
'''


import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
import csv
from time import time
import multiprocessing
import datetime
import os
import sys
import shutil
import numpy as np
import random



print("Kindly select one of the 3 Standard Problems you would like to solve on this run:")
print("For Ackleys Function: Key in A")
print("For Rosenbrock Function: Key in R")
print("For Shekel Function: Key in S")
std_problem = input ("Please key in a value: ")


if std_problem in 'A' or 'a':
    std_problem = "Ackley's Function"
elif std_problem in 'R' or 'r':
    std_problem = "Rosenbrock Function"
elif std_problem in 'S' or 's':
    std_problem = "Shekel Function"
else:
    sys.exit("Kindly insert a valid key")




def evaluator(x):
    '''Evaluator function, returns fitness and responses values'''
    # give the normalized candidates values inside the real design space
    x = [10*i-5 for i in x]
    # calculate fitness
    f = (sum([math.pow(i,4)-16*math.pow(i,2)+5*i for i in x])/2)
    # calculate values for other responses
    res = {'r1':f-5,'r2':2*f}
    fitness = dict(Obj=f,**res)
    return fitness


def mp_evaluator(x):
    '''Multiprocessing evaluation'''
    # ste number of cpus
    nprocs = 2
    # create pool
    pool = multiprocessing.Pool(processes=nprocs)
    results = [pool.apply_async(evaluator,[c]) for c in x]
    pool.close()
    pool.join()
    f = [r.get()['Obj'] for r in results]
    for r in results:
        del r.get()['Obj']
    # maximization or minimization problem
    maximize = False
    return (f, [r.get() for r in results],maximize)


def ackley_function(x1, x2):
    # returns the point value of the given coordinate
    part_1 = -0.2 * math.sqrt(0.5 * (x1 * x1 + x2 * x2))
    part_2 = 0.5 * (math.cos(2 * math.pi * x1) + math.cos(2 * math.pi * x2))
    value = math.exp(1) + 20 - 20 * math.exp(part_1) - math.exp(part_2)
    # returning the value
    return value


def ackley_function_range(x_range_array):
    # returns an array of values for the given x range of values
    value = np.empty([len(x_range_array[0])])
    for i in range(len(x_range_array[0])):
        # returns the point value of the given coordinate
        part_1 = -0.2 * math.sqrt(
            0.5 * (x_range_array[0][i] * x_range_array[0][i] + x_range_array[1][i] * x_range_array[1][i]))
        part_2 = 0.5 * (math.cos(2 * math.pi * x_range_array[0][i]) + math.cos(2 * math.pi * x_range_array[1][i]))

        value_point = math.exp(1) + 20 - 20 * math.exp(part_1) - math.exp(part_2)
        value[i] = value_point
    # returning the value array
    return value

def initialize(ants,var):
    '''Create initial solution matrix'''
    X = np.random.uniform(low=0,high=1,size=(ants,var))
    return X


def init_observer(filename,matrix,parameters,responses):
    '''Initial population observer'''
    p = []
    r = []
    f = []
    res = ['{0:>10}'.format(i)[:10] for i in responses]
    par = ['{0:>10}'.format(i)[:10] for i in parameters]
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    iteration = 0

    filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format('Iteration',', '.join(map(str, par)),'Fitness',', '.join(map(str, res))))

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))



def iter_observer(filename,matrix,parameters,responses,iteration):
    '''Iterations observer'''
    p = []
    r = []
    f = []
    for i in range(len(matrix)):
        p.append(matrix[i][0:len(parameters)])
        r.append(matrix[i][len(parameters):-1])
        f.append(matrix[i][-1])
    r = np.array(r)
    p = np.array(p)

    for i in range(len(r)):
        r[i] = ['{0:>10}'.format(r[i][j])[:10] for j in range(len(responses))]

    for i in range(len(p)):
        p[i] = ['{0:>10}'.format(p[i][j])[:10] for j in range(len(parameters))]

    f = ['{0:>10}'.format(i)[:10] for i in f]

    for i in range(len(matrix)):
        filename.write('{0:>10}, {1}, {2:>10}, {3}\n'.format(iteration,', '.join(map(str, p[i])),f[i],', '.join(map(str, r[i]))))


def correct_par(filename,par):
    """Replace normalized values with real"""
    columns = defaultdict(list)
    with open(filename) as f:
        reader = csv.DictReader(f,skipinitialspace=True)
        for row in reader:
            for (k,v) in row.items():
                columns[k].append(v)
        keys = columns.keys()
        for p in par:
            if p in keys:
                col = []
                for i,k in enumerate(columns[p]):
                    k = float(k)
                    if p in par:
                        n = 10*k-5
                    col.append(n)
                columns[p] = col

    outputfile = filename

    file = open(outputfile,'w+')
    head = []
    head.append('Iteration')
    for i in par:
        head.append(i)
    head.append('Fitness')
    for i in keys:
        if i not in head:
            head.append(i)
    par = ['{0:>10}'.format(i)[:10] for i in par]
    line = ['{0:>10}'.format(l)[:10] for l in head]
    file.write('{0}\n'.format(', '.join(map(str, line))))
    for i in range(len(columns.get('Iteration'))):
        line = []
        for j in head:
            line.append(columns.get(j)[i])
        line = ['{0:>10}'.format(l)[:10] for l in line]
        file.write('{0}\n'.format(', '.join(map(str, line))))
    file.close()


def formatTD(td):
    """ Format time output for report"""
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return '%s days %s h %s m %s s' % (days, hours, minutes, seconds)

def evolve(display):
    '''Executes the optimization'''


    start_time = time()

    # number of variables
    parameters_v = ['x1', 'x2']
    response_v = ['r1','r2']

    # create output file
    projdir = os.getcwd()
    if std_problem in "Rosenbrock Function":
        ind_file_name = '{0}/RosenbrockResults.csv'.format(projdir)
    elif std_problem in "Ackley's Function":
        ind_file_name = '{0}/AckleyResults.csv'.format(projdir)
    elif std_problem in "Shekel Function":
        ind_file_name = '{0}/Shekel_Results.csv'.format(projdir)
    else:
        ind_file_name = '{0}/results.csv'.format(projdir)
    ind_file = open(ind_file_name, 'w')

    # number of variables
    nVar = len(parameters_v)
    # size of solution archive
    nSize = 1000
    # number of ants
    nAnts = 10

    # parameter q
    q = 0.3

    # standard deviation
    qk = q*nSize

    # parameter xi (like pheromone evaporation)
    xi = 0.85

    # maximum iterations
    maximumiterations = 500
    # tolerance
    errormin = 0.01

    # bounds of variables
    Up = [1]*nVar
    Lo = [0]*nVar

    # initilize matrices
    S = np.zeros((nSize,nVar))
    S_f = np.zeros((nSize,1))

    plt.figure()


    # initialize the solution table with uniform random distribution and sort it
    print ('-----------------------------------------')
    print ('Starting initilization of solution matrix for ', std_problem)
    print ('-----------------------------------------')

    Srand = initialize(nSize,nVar)
    f,S_r,maximize = mp_evaluator(Srand)

    S_responses = []

    for i in range(len(S_r)):
        S_f[i] = f[i]
        k = S_r[i]
        row = []
        for r in response_v:
            row.append(k[r])
        S_responses.append(row)

    # add responses and "fitness" column to solution
    S = np.hstack((Srand,S_responses,S_f))
    # sort according to fitness (last column)
    S = sorted(S, key=lambda row: row[-1],reverse = maximize)
    S = np.array(S)

    init_observer(ind_file,S,parameters_v,response_v)

    # initilize weight array with pdf function

    '''if'''
    w = np.zeros((nSize))
    for i in range(nSize):
        if std_problem in "Rosenbrock Function":
            int_eq = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))
        elif std_problem in "Ackley's Function":
            int_eq = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))
        elif std_problem in "Shekel Function":
            int_eq = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))
        else:
            int_eq = 1/(qk*2*math.pi)*math.exp(-math.pow(i,2)/(2*math.pow(q,2)*math.pow(nSize,2)))
        w[i] = int_eq


    if display:
        x = []
        y = []
        for i in S:
            x.append(i[0])
            y.append(i[1])

        plt.scatter(x,y)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.pause(2)
        plt.cla()

    # initialize variables
    iterations = 1
    best_par = []
    best_obj = []
    best_sol = []
    best_res = []
    worst_obj = []
    best_par.append(S[0][:nVar])
    best_obj.append(S[0][-1])
    best_sol.append(S[0][:])
    best_res.append(S[0][nVar:-1])
    worst_obj.append(S[-1][-1])

    stop_condition = 0

    # iterations
    while True:
        print ('-----------------------------------------')
        print ('Iteration', iterations)
        print ('-----------------------------------------')
        print(best_sol)
        # choose Gaussian function to compose Gaussian kernel
        p = w/sum(w)

        # find best and index of best
        max_prospect = np.amax(p)
        ix_prospect = np.argmax(p)
        selection = ix_prospect

        # calculation of G_i
        # find standard deviation sigma
        sigma_s = np.zeros((nVar,1))
        sigma = np.zeros((nVar,1))
        for i in range(nVar):
            for j in range(nSize):
                sigma_s[i] = sigma_s[i] + abs(S[j][i] - S[selection][i])
            sigma[i] = xi / (nSize -1) * sigma_s[i]


        Stemp = np.zeros((nAnts,nVar))
        ffeval = np.zeros((nAnts,1))
        res = np.zeros((nAnts,len(response_v)))
        for k in range(nAnts):
            for i in range(nVar):
                Stemp[k][i] = sigma[i] * np.random.random_sample() + S[selection][i]
                if Stemp[k][i] > Up[i]:
                    Stemp[k][i] = Up[i]
                elif Stemp[k][i] < Lo[i]:
                    Stemp[k][i] = Lo[i]
        f,S_r,maximize = mp_evaluator(Stemp)

        S_f = np.zeros((nAnts,1))
        S_responses = []

        for i in range(len(S_r)):
            S_f[i] = f[i]
            k = S_r[i]
            row = []
            for r in response_v:
                row.append(k[r])
            S_responses.append(row)

        # add responses and "fitness" column to solution
        Ssample = np.hstack((Stemp,S_responses,S_f))

        # add new solutions in the solutions table
        Solution_temp = np.vstack((S,Ssample))

        # sort according to "fitness"
        Solution_temp = sorted(Solution_temp, key=lambda row: row[-1],reverse = maximize)
        Solution_temp = np.array(Solution_temp)

        # keep best solutions
        S = Solution_temp[:nSize][:]

        # keep best after each iteration
        best_par.append(S[0][:nVar])
        best_obj.append(S[0][-1])
        best_res.append(S[0][nVar:-1])
        best_sol.append(S[0][:])
        worst_obj.append(S[-1][-1])

        iter_observer(ind_file,S,parameters_v,response_v,iterations)

        if display:
            # plot new table
            x = []
            y = []
            for i in S:
                x.append(i[0])
                y.append(i[1])

            plt.scatter(x,y)
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.pause(2)

        if iterations > 1:
            diff = abs(best_obj[iterations]-best_obj[iterations-1])
            if diff <= errormin:
                stop_condition += 1

        iterations += 1
        if iterations > maximumiterations or stop_condition > 5:
            break
        else:
            if display:
                plt.cla()

    ind_file.close()

    total_time_s = time() - start_time
    total_time = datetime.timedelta(seconds=total_time_s)
    total_time = formatTD(total_time)

    # fix varibales values in output file
    correct_par(ind_file_name,parameters_v)

    best_sol = sorted(best_sol, key=lambda row: row[-1],reverse = maximize)

    print ("Best individual:", parameters_v)
    print (best_sol[0][0:len(parameters_v)])
    print ("Fitness:")
    print (best_sol[0][-1])
    print ("Responses:", response_v)
    print (best_sol[0][len(parameters_v):-1])


# Executes optimization run.
# If display = True plots ants in 2D design space
evolve(display = True)