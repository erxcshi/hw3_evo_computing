from evo import Evo
import random as rnd
import numpy as np
import pandas as pd
from process import *

import numpy as np

# constants - these can be in another module, but for the sake of this project they are here
# maps availabilities to integer
avail_dict = {'U': 3, 'W': 1, 'P': 0}

avail_converter = lambda x: int(avail_dict[x.decode()])

# load section times
sec_times = np.loadtxt("sections.csv",
                    delimiter=",", skiprows = 1, usecols = 2, dtype = 'str')

# load test values
test1 = arr = np.loadtxt("test1.csv",
                     delimiter=",")


time_dict = {val:idx+1 for idx, val in enumerate({x: None for x in sec_times}.keys())}

# load preferences using avail_dict as converter dict
A = np.loadtxt("tas.csv",
                     delimiter=",", skiprows = 1, usecols = range(3,20),converters = avail_converter)

# load times using sec times as constant
T = np.vectorize(lambda x: time_dict[x])(sec_times)

# load tas
O = np.loadtxt("tas.csv",
                     delimiter=",", skiprows = 1, usecols = 2)
# laod sections
S  = np.loadtxt("sections.csv",
                    delimiter=",", skiprows = 1, usecols =6)

# calculate objectives using functions that analyze constant arrays and sol arrays (G)
def unwilling(G):
    return (np.add(A, G) == 4).sum()
def unprefered(G):
    return (np.add(A, G) == 2).sum()
def conflicts(G):
    return sum([1 for row in G * T if sum(np.unique(row)) != sum(row)])
def overallocation(G):
    return sum(y-x for x,y in zip(O, G.sum(axis=1)) if x-y < 0)
def under_support(G):
    return sum(x-y for x,y in zip(S, G.sum(axis=0)) if x-y > 0)

# agents
def min_unwilling(solutions):
    # minimize unwilling by subtracting all assinged unwilling, always makes unwilling = 0
    L = solutions[0]
    conflicts = np.add(A, solutions) == 4
    conf_bin = conflicts * -1
    return (L + conf_bin)[0]
def min_overallocation(solutions):
    # randomly assign a TA to their max capability (n = max capability, sections are random)
    L = solutions[0]
    TA = rnd.randrange(0,43)
    arr = sorted(np.concatenate((np.ones((1, int(O[TA]))),np.zeros((1, 17 - int(O[TA])))), axis = None), key = lambda k: rnd.random())
    L[TA] = arr
    return L
def min_undersupport(solutions):
    # randomly assign TAS to a certain section (n = section min, tas are random)
    L = solutions[0]
    sec = rnd.randrange(0,17)
    arr = np.array(sorted(np.concatenate((np.ones((int(S[sec] + 2), 1)),np.zeros((41 - int(S[sec]), 1))), axis = None), key = lambda k: rnd.random())).reshape((43,))
    L[:, sec] = arr
    return L
def min_unpref(solutions):
    # minimize unprefered by substracting all assigned unprefered, always makes unprefered = 0
    L = solutions[0]
    conflicts = np.add(A, solutions) == 2
    conf_bin = conflicts * -1
    return (L + conf_bin)[0]

def get_best_sol(E):
    """ returns best solution from E population"""
    best_sol = [100, None]
    for metr in E.pop.keys():
        print([val[1] for val in metr])
        if max([val[1] for val in metr]) < best_sol[0]:
            best_sol[1] = metr
            best_sol[0] = max([val[1] for val in metr])
    return dict(E.pop.items())[metr], metr

def export_sols(E):
    """ exports best solution to csv"""
    d = {'overallocation': [], 'conflicts': [], 'undersupport': [], 'unwilling': [], 'unpreffered': []}
    for metr in E.pop.keys():
        for i in range(5):
            d[list(d.keys())[i]].append([val[1] for val in metr][i])
    dg = {'groupname' : ['final' * (len(d['unwilling'])+10)]}
    dg.update(d)
    df = pd.DataFrame(data=d)
    df.to_csv("final_solutions.csv", index=False)

def export_best(E):
    """ exports bes tsolution to csv"""
    header = np.loadtxt("sections.csv",
                delimiter=",", dtype = 'str')[:, 0]
    header = ','.join(header)
    indexes  = np.loadtxt("tas.csv",
                 delimiter=",", skiprows = 1, usecols = 0).reshape((43,1))
    np.savetxt('best_soloution.csv', np.append( indexes,get_best_sol(E)[0], 1), delimiter=',', fmt='%.0f', header = header)


def main():


    E = Evo()

    # add fitness criteria
    E.add_fitness_criteria("over", overallocation)
    E.add_fitness_criteria("con", conflicts)
    E.add_fitness_criteria("under", under_support)
    E.add_fitness_criteria("unw", unwilling)
    E.add_fitness_criteria("unp", unprefered)

    # add agents
    E.add_agent("over", min_overallocation, 1)
    E.add_agent("unw", min_unwilling, 1)
    E.add_agent("under", min_undersupport, 1)
    E.add_agent("unp", min_unpref, 1)

    # make random solutiom 1
    rand = rnd.uniform(0.6, 1)
    L = np.random.choice([0, 1], size=731, p=[rand, 1 - rand]).reshape((43, 17))
    E.add_solution(L)

    # evolve one million times
    E.evolve(2000)

    export_sols(E)
    export_best(E)
    print(get_best_sol(E))

main()
