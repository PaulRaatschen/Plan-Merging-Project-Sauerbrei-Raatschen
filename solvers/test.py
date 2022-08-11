from os import path
from iterative_solving import IterativeSolver
from prioritized_planning import PrioritizedPlanningSolver
from solution import Solution
import logging

"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
INSTANCE_FILE : str = path.join(WORKING_DIR,r'..\naiveplanning\instances\sideways_parking.lp')

sol : Solution = PrioritizedPlanningSolver(INSTANCE_FILE,log_level=logging.DEBUG,backtrack=True).solve()

print(sol.satisfied)
print(sol.execution_time)

sol.save(r'sol.lp')






