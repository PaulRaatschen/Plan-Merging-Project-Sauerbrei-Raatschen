from os import path
from iterative_solving import IterativeSolver
from prioritized_planning import PrioritizedPlanningSolver
from solution import Solution
import logging

from cbs import CBSSolver

"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
INSTANCE_FILE : str = path.join(WORKING_DIR,r'..\naiveplanning\instances\dodge_hell.lp')

sol : Solution = CBSSolver(INSTANCE_FILE,log_level=logging.DEBUG).solve()

print(sol.satisfied)
print(f'Execution time : {sol.execution_time:.3}s\n sum of costs : {sol.get_soc()}\n normalized : {sol.get_norm_soc():.3}\n makespan: {sol.get_makespan()}\n normalized : {sol.get_norm_makespan():.3}\n total moves : {sol.get_total_moves()}\n normalized : {sol.get_norm_total_moves():.3}\n density : {sol.get_density():.3}')

sol.save(r'sol.lp')






