import copy
import random as rnd
from functools import reduce
import statistics

class Evo:
    def __init__(self):
        self.pop = {}
        self.fitness = {}
        self.agents = {}
        self.generations = []

    def add_fitness_criteria(self, name, f):
        """ set fitness criteria """
        self.fitness[name] = f

    def add_agent(self, name, op, k=1):
        """ set agents """
        self.agents[name] = (op, k)

    def add_solution(self, sol):
        """ add each new solution """
        eval = tuple([(name, f(sol)) for name, f in self.fitness.items()])
        self.pop[eval] = sol

    def get_random_solutions(self, k=1):
        """ get a random solution """
        popvals = tuple(self.pop.values())
        return [copy.deepcopy(rnd.choice(popvals)) for _ in range(k)]

    def run_agent(self, name):
        """ run a specific agent """
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)

    @staticmethod
    def _dominates(p,q):
        """ soltuion p dominates solution q if it is equal or better in all objectives """
        pscores = [score for _,score in p]
        qscores = [score for _,score in q]
        score_diffs = list(map(lambda x,y: int(y-x), pscores, qscores))

        min_diff = min(score_diffs)
        max_diff = max(score_diffs)

        return min_diff >= 0.0 and max_diff > 0.0

    @staticmethod
    def _reduce_nds(S, p):
        """ helper function, removes non dominants from dict"""
        return S - {q for q in S if Evo._dominates(p,q)}

    def remove_dominated(self):
        """ uses reduce helper on population """
        nds = reduce(Evo._reduce_nds, self.pop.keys(), self.pop.keys())
        self.pop = {k:self.pop[k] for k in nds}

    def evolve(self, n=40, dom=5):
        """ apply random agents through generations """
        agent_names = list(self.agents.keys())
        for i in range(n):

            pick = rnd.choice(agent_names)
            self.run_agent(pick)

            if i % dom == 0:
                self.remove_dominated()
            self.remove_dominated()

    def __str__(self):
        rslt = ''

        for eval, sol in self.pop.items():
            rslt += str(dict(eval)) + ":\t" + str(sol) + "\n"

        return rslt
