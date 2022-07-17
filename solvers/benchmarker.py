from argparse import ArgumentParser
from asyncio.windows_events import NULL
from os import path
import pandas as pd
import prioritized_planning
import sequential_planning
import cbs_solver
from solution import Solution 

parser : ArgumentParser = ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("-tag", type=int,default = 0)
args = parser.parse_args()
instanceName = path.basename(args.instance)

ppSolution = prioritized_planning.PrioritizedPlanningSolver(args.instance,False,False,10,NULL).solve()
spSolution = sequential_planning.benchmark(args.instance).result
cbsSolution = cbs_solver.CBS_Solver(args.instance,False,NULL).solve()

ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'max_horizon': [ppSolution.max_horizon],'cost' : [ppSolution.cost],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["SP"],'max_horizon': [spSolution.max_horizon],'cost' : [spSolution.cost],'exec_time' : [spSolution.execution_time],'satisfied' : [spSolution.satisfied]})
cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS"],'max_horizon': [cbsSolution.max_horizon],'cost' : [cbsSolution.cost],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
combineddf = pd.concat([ppdf,spdf,cbsdf])

combineddf.to_csv('results.csv', mode='a', index=False, header = False)