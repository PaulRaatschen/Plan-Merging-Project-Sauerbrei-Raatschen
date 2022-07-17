from solvers.cbs_solver import CBS_Solver
from solvers.solution import Solution
from os import path

WDIR : str = path.abspath(path.dirname(__file__))

solver = CBS_Solver(path.join(WDIR,'test_instance.lp'))

solution = solver.solve()

solution.save('plan.lp')

