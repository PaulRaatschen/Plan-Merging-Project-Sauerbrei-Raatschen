from argparse import ArgumentParser
import logging
from os import path
import sys
import pandas as pd
from prioritized_planning import PrioritizedPlanningSolver
from iterative_solving import IterativeSolver
from cbs import CBSSolver
from solution import Solution
from generate_instance import GenerateInstance

'''Logging setup'''
parser : ArgumentParser = ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("-tag", type=int,default = 0)
parser.add_argument("-gi", "--GenerateInstance", default=False, action="store_true")
args = parser.parse_args()
instanceName = path.basename(args.instance)


def save_instance_info(name : str, solution : Solution):
    xSize : int = 0
    ySize : int = 0
    nodecount : int = 0
    for atom in solution.instance_atoms:
        if atom.arguments[0].arguments[0].name=='node':
            nodecount += 1
            if xSize < atom.arguments[1].arguments[1].arguments[0].number: xSize = atom.arguments[1].arguments[1].arguments[0].number
            if ySize < atom.arguments[1].arguments[1].arguments[1].number: ySize = atom.arguments[1].arguments[1].arguments[1].number
        

    instance = pd.DataFrame({'instance': [name],'Xsize':xSize,'Ysize':ySize,'blocked_nodes': xSize*ySize-nodecount,'density' : solution.get_density(), 'number_of_agents': len(solution.agents)})
    instance.to_csv(f'{name}.csv', mode='a', index=False, header = not path.exists(f'{name}.csv'))





if(args.GenerateInstance == False):

    print("PrioritizedPlanning-Start")

    ppSolution : Solution = PrioritizedPlanningSolver(args.instance,backtrack=True,log_level=logging.CRITICAL).solve()
    ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'soc' : [ppSolution.cost],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
    ppdf.to_csv('results.csv', mode='a', index=False, header = not path.exists('results.csv'))

    save_instance_info(instanceName,ppSolution)

    print("IterativeSolving-Start")
    spSolution : Solution = IterativeSolver(args.instance,log_level=logging.CRITICAL) 
    spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["SP"],'makespan': [spSolution.get_makespan()],'soc' : [spSolution.cost],'exec_time' : [spSolution.execution_time],'satisfied' : [spSolution.satisfied]})
    spdf.to_csv('results.csv', mode='a', index=False, header = False)


    print("CBS-Start")
    cbsSolution : Solution = CBSSolver(args.instance,log_level=logging.CRITICAL).solve()
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS"],'max_horizon': [cbsSolution.get_makespan()],'soc' : [cbsSolution.cost],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

else:
    for i in range(1,5):
        print("PrioritizedPlanning-Start")

        ppSolution : Solution = PrioritizedPlanningSolver(args.instance,backtrack=True,log_level=logging.CRITICAL).solve()
        ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'soc' : [ppSolution.cost],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
        ppdf.to_csv('results.csv', mode='a', index=False, header = not path.exists('results.csv'))

        save_instance_info(instanceName,ppSolution)

        print("IterativeSolving-Start")
        spSolution : Solution = IterativeSolver(args.instance,log_level=logging.CRITICAL) 
        spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["SP"],'makespan': [spSolution.get_makespan()],'soc' : [spSolution.cost],'exec_time' : [spSolution.execution_time],'satisfied' : [spSolution.satisfied]})
        spdf.to_csv('results.csv', mode='a', index=False, header = False)


        print("CBS-Start")
        cbsSolution : Solution = CBSSolver(args.instance,log_level=logging.CRITICAL).solve()
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS"],'max_horizon': [cbsSolution.get_makespan()],'soc' : [cbsSolution.cost],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

