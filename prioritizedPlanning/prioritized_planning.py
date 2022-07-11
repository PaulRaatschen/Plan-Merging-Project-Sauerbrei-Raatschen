from __future__ import annotations
from typing import Callable, Dict, List, Set, Tuple, Union
from clingo import Control, Number, Function, Symbol, Model
from time import perf_counter
from math import inf
from os import path
from argparse import ArgumentParser, Namespace
from sys import stdout
import logging

"""Logging setup"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(message)s')
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

"""Command line argument parsing"""
parser : ArgumentParser = ArgumentParser()
parser.add_argument("instance", type=str)
parser.add_argument("-b", "--benchmark", default=False, action="store_true")
parser.add_argument("-o", "--optimize", default=False, action="store_true")
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--backtrack",default=False, action="store_true")
parser.add_argument("--maxdepth",default=10,type=int)
parser.add_argument("--debug", default=False, action="store_true")
args : Namespace = parser.parse_args()

if args.debug:
    logger.setLevel(logging.DEBUG)

"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')
PREPROCESSING_FILE : str = path.join(ENCODING_DIR,'setup.lp')
SAPF_FILE : str = path.join(ENCODING_DIR,'singleAgentPF_inc.lp')
CONFLICT_DETECTION_FILE : str = path.join(ENCODING_DIR,'conflict_detection.lp')
SAPF_NC_FILE : str = path.join(ENCODING_DIR,'singleAgentPF_nc_inc.lp')
INSTANCE_FILE : str = args.instance

class Solution:

    def __init__(self,agents : List[int] = None,inits : List[Symbol] = None, instance_atoms : List[Symbol] = None, plans : Dict[int,Dict[str,Union[List[Symbol],int]]] = None, max_horizon : int = 0):
        self.agents = agents if agents else []
        self.inits = inits if inits else []
        self.instance_atoms = instance_atoms if instance_atoms else []
        self.plans = plans if plans else {}
        self.max_horizon = max_horizon
        self.cost : int = 0

    def clear_plans(self) -> None:
        self.plans = {}

    def clear_plan(self,agent) -> None:
        if agent in self.plans:
            self.plans[agent] = {'occurs' : [],'positions' : [], 'cost' : inf}

def preprocessing() -> Solution:

    solution : Solution = Solution()

    def preprocessing_parser(model : Model, solution : Solution) -> bool:

        for atom in model.symbols(atoms=True):
            if(atom.name == 'init'):
                solution.inits.append(atom)
            elif(atom.name == 'numOfRobots'):
                solution.agents = list(range(1,atom.arguments[0].number+1))
            elif(atom.name == 'numOfNodes'):
                solution.max_horizon = atom.arguments[0].number * 2
            else:
                solution.instance_atoms.append(atom)

        return False

    logger.debug("Preprocessing invoked")

    ctl : Control = Control(["-Wnone"])

    ctl.load(INSTANCE_FILE)

    ctl.load(PREPROCESSING_FILE)

    ctl.ground([("base",[])])

    ctl.solve(on_model=lambda model : preprocessing_parser(model,solution))

    return solution

def optimize_schedule(solution : Solution) -> List[int]:

    logger.debug("Optimization invoked")

    ctl : Control
    plan_lenghts : List[int] = [0] * len(solution.agents)
    positions : List[Symbol] = []
    schedule : List[int] = []

    def optimization_parser(model : Model, schedule : List[int]) -> bool:

        agent_conflicts : List[Tuple(int,int)] = []

        for atom in model.symbols(shown=True):
            agent_conflicts.append((int(atom.arguments[0])),int(atom.arguments[1]))

        if agent_conflicts:
            schedule.extend(sorted(agent_conflicts, key=lambda item : item[1])[0])

        return False

    for agent in solution.agents:

        ctl = Control(arguments=['-Wnone',f'-c r={agent}'])

        ctl.load(SAPF_NC_FILE)

        with ctl.backend() as backend:
            for atom in solution.instance_atoms:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

        plan_lenghts[agent-1] = incremental_solving(ctl,solution.max_horizon,lambda model : positions.extend(model.symbols(shown=True)))

        logger.debug(f'Optimal Plan for agent {agent} with cost {plan_lenghts[agent-1]}')

    ctl = Control(arguments=['-Wnone'])

    ctl.load(CONFLICT_DETECTION_FILE)

    with ctl.backend() as backend:
        for plan in solution.plans:
            for position in plan['position']:
                fact = backend.add_atom(position)
                backend.add_rule([fact])

    ctl.ground([("base",[])])

    ctl.solve(on_model=lambda model : optimization_parser(model,schedule))

    return schedule

    
def incremental_solving(ctl : Control, max_horizon : int, model_parser : Callable[[Model],bool]) -> int:

    ret, step = None, 0

    while((step < max_horizon) and (ret is None or (step < max_horizon and not ret.satisfiable))):
        parts = []
        parts.append(("check", [Number(step)]))
        if step > 0:
            ctl.release_external(Function("query", [Number(step - 1)]))
            parts.append(("step", [Number(step)]))
        else:
            parts.append(("base", []))
        ctl.ground(parts)
        ctl.assign_external(Function("query", [Number(step)]), True)
        ret, step = ctl.solve(on_model=model_parser), step + 1   

    return inf  if not ret or (not ret.satisfiable) else step - 1


def plan_path(agent : int, solution : Solution) -> bool:

    ctl : Control
    cost : int
    old_cost : int = 0

    if agent in solution.plans:
        old_cost = solution.plans[agent]['cost'] 

    logger.debug(f'Planning for agent {agent}')

    solution.plans[agent] = {'occurs' : [],'positions' : [], 'cost' : inf}

    def plan_path_parser(model : Model, agent : int, solution : Solution):
        for atom in model.symbols(shown=True):
                if(atom.name == 'occurs'):
                    solution.plans[agent]['occurs'].append(atom)
                else:
                    solution.plans[agent]['positions'].append(atom)
        return False

    ctl = Control(arguments=['-Wnone',f'-c r={agent}'])

    ctl.load(SAPF_FILE)

    with ctl.backend() as backend:
        for atom in solution.instance_atoms:
            fact = backend.add_atom(atom)
            backend.add_rule([fact])

        for plan in solution.plans.values():
            for position in plan['positions']:
                fact = backend.add_atom(position)
                backend.add_rule([fact])

    cost = incremental_solving(ctl,solution.max_horizon,lambda model : plan_path_parser(model,agent,solution))

    solution.plans[agent]['cost'] = cost
    solution.cost += (cost-old_cost)

    logger.debug(f'Planning finished with cost {cost}')

    return cost < inf


def main() -> None:

    logger.debug("Programm started")
    
    solution : Solution = preprocessing()
    ordering : List[int] = solution.agents
    orderings : Set[Tuple[int]] = set(ordering)
    finished_index : int = 0 
    t_start : float = perf_counter()
    t_end : float
    max_depth : int = args.maxdepth
    satisfied : bool


    
    if args.optimize:
        schedule : List[int] = optimize_schedule(solution)

        if schedule:
            solution.agents = schedule

        solution.clear_plans()
    
    while max_depth > 0:

        satisfied = True

        for index, agent in  enumerate(ordering[finished_index:]):

            if not plan_path(agent,solution):
                satisfied = False
                logger.debug(f'No solution found for agent {agent}')

                if args.backtrack and index > 0:
                    finished_index = index - 1
                    agent_to_swap : int = ordering[index-1]
                    ordering[index-1] = ordering[index]
                    ordering[index] = agent_to_swap
                    if tuple(ordering) in orderings:
                        logger.debug("Repetition in oderings")
                        max_depth = 1
                    else:
                        orderings.add(tuple(ordering))
                    solution.clear_plan(agent_to_swap)
                    max_depth -= 1
                    break

        if not args.backtrack or satisfied:
            break

    logger.info("Planning finished")

    if satisfied:
        logger.info("Global solution found")
    else:
        logger.info("No global solution found")

    if args.benchmark:
        t_end = perf_counter()
        logger.info(f'Execution time : {t_end-t_start:.2f}s')
        logger.info(f'Total model cost : {solution.cost}')
    
    with open('plan.lp','w') as file:
        for init in solution.inits:
            file.write(f'{init}.')
        for plan in solution.plans.values():
            for occur in plan['occurs']:
                file.write(f'{occur}.')


if __name__ == "__main__":
    main()