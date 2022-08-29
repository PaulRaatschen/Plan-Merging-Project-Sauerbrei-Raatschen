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
cbs_timeout : int = 999999999
isSolution : Solution
ppSolution : Solution
cbsSolution : Solution
FILENAME : str = 'results.csv' 


def save_instance_info(name : str, solution : Solution):
    x_size : int = 0
    y_size : int = 0
    for atom in solution.inits:
        if atom.arguments[0].arguments[0].name=='node':
            if x_size < atom.arguments[1].arguments[1].arguments[0].number: x_size = atom.arguments[1].arguments[1].arguments[0].number
            if y_size < atom.arguments[1].arguments[1].arguments[1].number: y_size = atom.arguments[1].arguments[1].arguments[1].number
        

    instance = pd.DataFrame({'instance': [name],'xsize':x_size,'ysize':y_size,'blocked_nodes': x_size*y_size-solution.num_of_nodes,'density' : solution.get_density(), 'number_of_agents': len(solution.agents)})
    instance.to_csv('instances.csv', mode='a', index=False, header = not path.exists(f'{name}.csv'))





if(args.GenerateInstance == False):

    print("IterativeSolving-Start")
    itSolution = IterativeSolver(args.instance,log_level=logging.CRITICAL).solve()
    initial_plans = itSolution.get_initial_plans()
    spdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["IS"],'makespan': [itSolution.get_makespan()],'norm_makespan': [itSolution.get_norm_makespan()],'soc' : [itSolution.get_soc()],'norm_soc' : [itSolution.get_norm_soc()],'total_moves' : [itSolution.get_total_moves()],'norm_total_moves' : [itSolution.get_norm_total_moves()],'exec_time' : [itSolution.execution_time],'satisfied' : [itSolution.satisfied]})
    spdf.to_csv(FILENAME, mode='a', index=False, header = not path.exists(FILENAME))

    save_instance_info(instanceName,itSolution)

    print("PrioritizedPlanning-Start")

    ppSolution = PrioritizedPlanningSolver(args.instance,backtrack=True,log_level=logging.CRITICAL).solve()
    ppSolution.initial_plans = initial_plans
    ppdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'norm_makespan': [ppSolution.get_norm_makespan()],'soc' : [ppSolution.get_soc()],'norm_soc' : [ppSolution.get_norm_soc()],'total_moves' : [ppSolution.get_total_moves()],'norm_total_moves' : [ppSolution.get_norm_total_moves()],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
    ppdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("CBS-Soc-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("GICBS-Soc-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,greedy=True,icbs=True,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["GICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("ICBS-Soc-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["ICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("MCBS-Soc-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,meta=True,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MCBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("MICBS-Soc-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,meta=True,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)    

    print("CBS-Ms-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,makespan=True,timeout=cbs_timeout).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["CBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("GICBS-Ms-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,greedy=True,icbs=True,timeout=cbs_timeout,makespan=True).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["GICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("ICBS-Ms-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,timeout=cbs_timeout,makespan=True).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["ICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("MCBS-Ms-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,meta=True,timeout=cbs_timeout,makespan=True).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MCBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

    print("MICBS-Ms-Start")
    cbsSolution = CBSSolver(args.instance,log_level=logging.CRITICAL,icbs=True,meta=True,timeout=cbs_timeout,makespan=True).solve()
    cbsSolution.initial_plans = initial_plans
    cbsdf = pd.DataFrame({'instance': [instanceName],'tag':[args.tag],'solver':["MICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
    cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)  

else:
    for complexity in range(1,9):

        print("Complexity: " + str(complexity))

        for iteration in range(1,5):
            print("     Iteration:" + str(iteration))
            modinstanceName = instanceName + str(complexity) + "." + str(iteration)
            GenerateInstance.createInstance(5,5,complexity,"Random",5)


            print("         IterativeSolving-Start")
            itSolution = IterativeSolver("generatedInstance.lp",log_level=logging.CRITICAL).solve()
            initial_plans = itSolution.get_initial_plans()
            spdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["IS"],'makespan': [itSolution.get_makespan()],'norm_makespan': [itSolution.get_norm_makespan()],'soc' : [itSolution.get_soc()],'norm_soc' : [itSolution.get_norm_soc()],'total_moves' : [itSolution.get_total_moves()],'norm_total_moves' : [itSolution.get_norm_total_moves()],'exec_time' : [itSolution.execution_time],'satisfied' : [itSolution.satisfied]})
            spdf.to_csv(FILENAME, mode='a', index=False, header = not path.exists(FILENAME))

            save_instance_info(instanceName,itSolution)

            print("         PrioritizedPlanning-Start")

            ppSolution = PrioritizedPlanningSolver("generatedInstance.lp",backtrack=True,log_level=logging.CRITICAL).solve()
            ppSolution.initial_plans = initial_plans
            ppdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["PP"],'makespan': [ppSolution.get_makespan()],'norm_makespan': [ppSolution.get_norm_makespan()],'soc' : [ppSolution.get_soc()],'norm_soc' : [ppSolution.get_norm_soc()],'total_moves' : [ppSolution.get_total_moves()],'norm_total_moves' : [ppSolution.get_norm_total_moves()],'exec_time' : [ppSolution.execution_time],'satisfied' : [ppSolution.satisfied]})
            ppdf.to_csv(FILENAME, mode='a', index=False, header = False)


            print("         CBS-Soc-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["CBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         GICBS-Soc-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,greedy=True,icbs=True,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["GICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         ICBS-Soc-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,icbs=True,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["ICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         MCBS-Soc-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,meta=True,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["MCBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         MICBS-Soc-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,icbs=True,meta=True,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["MICBS-SOC"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)    

            print("         CBS-Ms-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,makespan=True,timeout=cbs_timeout).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["CBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         GICBS-Ms-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,greedy=True,icbs=True,timeout=cbs_timeout,makespan=True).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["GICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         ICBS-Ms-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,icbs=True,timeout=cbs_timeout,makespan=True).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["ICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         MCBS-Ms-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,meta=True,timeout=cbs_timeout,makespan=True).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["MCBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False)

            print("         MICBS-Ms-Start")
            cbsSolution = CBSSolver("generatedInstance.lp",log_level=logging.CRITICAL,icbs=True,meta=True,timeout=cbs_timeout,makespan=True).solve()
            cbsSolution.initial_plans = initial_plans
            cbsdf = pd.DataFrame({'instance': [modinstanceName],'tag':[args.tag],'solver':["MICBS-MS"],'makespan': [cbsSolution.get_makespan()],'norm_makespan': [cbsSolution.get_norm_makespan()],'soc' : [cbsSolution.get_soc()],'norm_soc' : [cbsSolution.get_norm_soc()],'total_moves' : [cbsSolution.get_total_moves()],'norm_total_moves' : [cbsSolution.get_norm_total_moves()],'exec_time' : [cbsSolution.execution_time],'satisfied' : [cbsSolution.satisfied]})
            cbsdf.to_csv(FILENAME, mode='a', index=False, header = False) 