from argparse import ArgumentParser
import logging
from os import path
import sys
from typing import Dict
import pandas as pd
from prioritized_planning import PrioritizedPlanningSolver
from iterative_solving import IterativeSolver
from cbs import CBSSolver
from solution import Solution, Plan
from generate_instance import GenerateInstance

'''Command line argument parsing'''
parser : ArgumentParser = ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("-tag", type=int,default = 0)
parser.add_argument("-gi", "--GenerateInstance", default=False, action="store_true")
args = parser.parse_args()
instanceName : str = path.basename(args.instance)
initial_plans : Dict[int,Plan]
cbs_timeout : int = 300
isSolution : Solution
ppSolution : Solution
cbsSolution : Solution


def save_instance_info(name : str, solution : Solution):
    xSize : int = 0
    ySize : int = 0
    for atom in solution.instance_atoms:
        if atom.arguments[0].arguments[0].name=='node':
            if xSize < atom.arguments[1].arguments[1].arguments[0].number: xSize = atom.arguments[1].arguments[1].arguments[0].number
            if ySize < atom.arguments[1].arguments[1].arguments[1].number: ySize = atom.arguments[1].arguments[1].arguments[1].number
        

    instance = pd.DataFrame({'instance': [name],'xsize':xSize,'ysize':ySize,'blocked_nodes': xSize*ySize-solution.num_of_nodes,'density' : solution.get_density(), 'number_of_agents': len(solution.agents)})
    instance.to_csv(f'{name}.csv', mode='a', index=False, header = not path.exists(f'{name}.csv'))





if(args.GenerateInstance == False):

    print("IterativeSolving-Start")
    itSolution = IterativeSolver(args.instance,log_level=logging.CRITICAL)
    initial_plans = itSolution.get_initial_plans() 
    spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [itSolution.get_makespan()],'norm_makespan': [itSolution.get_norm_makespan()],'soc' : [itSolution.get_makespan()],'norm_soc' : [itSolution.get_norm_makespan()],'total_moves' : [itSolution.get_total_moves()],'norm_total_moves' : [itSolution.get_norm_total_moves()],'exec_time' : [itSolution.execution_time],'satisfied' : [itSolution.satisfied]})
    spdf.to_csv('results.csv', mode='a', index=False, header = False)

    save_instance_info(instanceName,itSolution)

    print("PrioritizedPlanning-Start")

    ppSolution = PrioritizedPlanningSolver(args.instance,backtrack=True,log_level=logging.CRITICAL).solve()
    ppSolution.initial_plans = initial_plans
    ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'norm_makespan': [ppSolution.get_norm_makespan()],'soc' : [ppSolution.get_makespan()],'norm_soc' : [ppSolution.get_norm_makespan()],'total_moves' : [ppSolution.get_total_moves()],'norm_total_moves' : [ppSolution.get_norm_total_moves()],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
    ppdf.to_csv('results.csv', mode='a', index=False, header = not path.exists('results.csv'))

    print("CBS-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

else:
    for i in range(1,5):

        print("IterativeSolving-Start")
        itSolution = IterativeSolver(args.instance,log_level=logging.CRITICAL) 
        initial_plans = itSolution.get_initial_plans()
        spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [itSolution.get_makespan()],'norm_makespan': [itSolution.get_norm_makespan()],'soc' : [itSolution.get_makespan()],'norm_soc' : [itSolution.get_norm_makespan()],'total_moves' : [itSolution.get_total_moves()],'norm_total_moves' : [itSolution.get_norm_total_moves()],'exec_time' : [itSolution.execution_time],'satisfied' : [itSolution.satisfied]})
        spdf.to_csv('results.csv', mode='a', index=False, header = False)

        save_instance_info(instanceName,itSolution)

        print("PrioritizedPlanning-Start")

        ppSolution = PrioritizedPlanningSolver(args.instance,backtrack=True,log_level=logging.CRITICAL).solve()
        ppSolution.initial_plans = initial_plans
        ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'norm_makespan': [ppSolution.get_norm_makespan()],'soc' : [ppSolution.get_makespan()],'norm_soc' : [ppSolution.get_norm_makespan()],'total_moves' : [ppSolution.get_total_moves()],'norm_total_moves' : [ppSolution.get_norm_total_moves()],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
        ppdf.to_csv('results.csv', mode='a', index=False, header = not path.exists('results.csv'))


        print("CBS-Start")
        cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,timeout=cbs_timeout).solve()
        cbsSolution.initial_plans = initial_plans
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

        print("GICBS-Start")
        cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,greedy=True,icbs=True,timeout=cbs_timeout).solve()
        cbsSolution.initial_plans = initial_plans
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["GICBS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

        print("ICBS-Start")
        cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,timeout=cbs_timeout).solve()
        cbsSolution.initial_plans = initial_plans
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["ICBS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

        print("MCBS-Start")
        cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,meta=True,timeout=cbs_timeout).solve()
        cbsSolution.initial_plans = initial_plans
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MCBS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

        print("MICBS-Start")
        cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,meta=True,timeout=cbs_timeout).solve()
        cbsSolution.initial_plans = initial_plans
        cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MICBS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_makespan()],'norm_soc' : [cbsSolution.get_norm_makespan()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
        cbsdf.to_csv('results.csv', mode='a', index=False, header = False)

