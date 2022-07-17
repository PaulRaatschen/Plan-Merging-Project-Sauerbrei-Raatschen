from __future__ import annotations
from bisect import insort
from typing import Dict, List, Tuple, Union
from clingo import Control, Number, Function, Symbol, Model
from clingo.solving import SolveResult
from time import perf_counter
from os import path
from argparse import ArgumentParser, Namespace
from sys import exit, stdout
from math import inf
import logging
from copy import deepcopy, copy
from solution import Solution


"""Logging setup"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(message)s')
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')
PREPROCESSING_FILE : str = path.join(ENCODING_DIR,'setup.lp')
SAPF_FILE : str = path.join(ENCODING_DIR,'singleAgentPF_inc.lp')
VALIADTION_FILE : str = path.join(ENCODING_DIR,'validate.lp')

class CTNode:
    """
    CTNode class represents a node in the constraint tree of the conflict based search algorithm.
    Stores solution atoms for each agent, contraints imposed on agents paths and the total cost of all paths

    Methods:
        __init__(self,plans : Dict[int,List[int,List[Symbol],List[Symbol]]]=None, constraints : List[Symbol]=None, atoms : List[Symbol]=None) -> None
        low_level(self,agent : int) -> bool
        validate_plans(self) -> List[Symbol]
        branch(self,conflict : Symbol) -> Tuple[CTNode,CTNode]    
    """

    def __init__(self,plans : Dict[int,Dict[str,Union[List[Symbol],List[Symbol],int]]]=None, constraints : List[Symbol]=None, atoms : List[Symbol]=None, cost : int = 0) -> None:
        self.plans = plans if plans else {}
        self.constraints = constraints if constraints else []
        self.atoms = atoms
        self.cost = cost

    def __gt__(self, other : CTNode) -> bool:
        if self.cost > other.cost:
            return True 
        elif self.cost == other.cost:
            return len(self.constraints) > len(other.constraints)
        else:
            return False

    def __ge__(self, other : CTNode) -> bool:
        if self.cost > other.cost:
            return True 
        elif self.cost == other.cost:
            return len(self.constraints) >= len(other.constraints)
        else:
            return False

    def __lt__(self, other : CTNode) -> bool:
        return not self.__ge__(other)

    def __le__(self, other : CTNode) -> bool:
        return not self.__gt__(other)

    def low_level(self,agent : int, horizon : int) -> bool:

        old_cost : int = 0
        agent_cost : int
        ctl : Control = Control(['-Wnone',f'-c r={agent}'])
        step : int = 0 
        ret : SolveResult = None

        def low_level_parser(model : Model, agent : int, plans : Dict[int,Dict[str,Union[List[Symbol],List[Symbol],int]]]) -> bool:
            for atom in model.symbols(shown=True):
                if(atom.name == 'occurs'):
                    plans[agent]['occurs'].append(atom)
                else:
                    plans[agent]['positions'].append(atom)
        
        logger.debug(f"low level search invoked for agent {agent}")

        if agent in self.plans:
            old_cost = self.plans[agent]['cost']

        self.plans[agent] = {'positions' : [], 'occurs' : [], 'cost' : 0}

        ctl.load(SAPF_FILE)

        with ctl.backend() as backend:
            for atom in self.atoms + self.constraints:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

        while ((step < horizon) and (ret is None or (not ret.satisfiable))):
                    parts : List[Tuple[str,List[Symbol]]] = []
                    parts.append(("check", [Number(step)]))
                    if step > 0:
                        ctl.release_external(Function("query", [Number(step - 1)]))
                        parts.append(("step", [Number(step)]))
                    else:
                        parts.append(("base", []))
                    ctl.ground(parts)
                    ctl.assign_external(Function("query", [Number(step)]), True)
                    ret, step = ctl.solve(on_model=lambda model : low_level_parser(model,agent,self.plans)), step + 1
                    
        agent_cost = inf if (ret and (not ret.satisfiable)) else step - 1

        self.plans[agent]['cost'] = agent_cost

        logger.debug(f"low level search terminated for agent {agent} with cost {agent_cost}")

        self.cost += (agent_cost-old_cost)

        return agent_cost < inf

    def validate_plans(self) -> Union[Symbol,bool]:

        ctl : Control
        conflicts : List[Symbol] = []

        def validate_parser( model : Model, conflicts : List[Symbol]) -> bool:
            for atom in model.symbols(shown=True):
                if(atom.name == 'minConflict'):
                    conflicts.append(atom)
            return False

        logger.debug("Validate plans invoked")

        ctl = Control(['-Wnone'])

        ctl.load(VALIADTION_FILE)

        with ctl.backend() as backend:
            for instance in self.plans.values():
                for atom in instance['positions']:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])

        ctl.ground([('base',[])])

        ctl.solve(on_model=lambda model : validate_parser(model,conflicts))

        return conflicts[0] if conflicts else False

    def branch(self,conflict : Symbol, max_horizon : int) -> Tuple[CTNode,CTNode]:

        logger.debug("branch invoked")

        node1 : CTNode = CTNode(deepcopy(self.plans),copy(self.constraints),self.atoms,self.cost)
        node2 : CTNode = CTNode(deepcopy(self.plans),copy(self.constraints),self.atoms,self.cost)

        conflict_type : str = conflict.arguments[0].name
        agent1 : Number = conflict.arguments[1]
        agent2 : Number  = conflict.arguments[2]
        time : Number = conflict.arguments[4]

        if(conflict_type == 'vertex'):
            loc : Function = conflict.arguments[3]
            node1.constraints.append(Function(name="constraint", arguments=[agent1,loc,time]))
            node2.constraints.append(Function(name="constraint", arguments=[agent2,loc,time]))

        else:
            loc1 : Function = conflict.arguments[3].arguments[0]
            loc2 : Function = conflict.arguments[3].arguments[1]
            move1 : Function = Function('',[Number(loc2.arguments[0].number-loc1.arguments[0].number),Number(loc2.arguments[1].number-loc1.arguments[1].number)],True)
            move2 : Function = Function('',[Number(loc1.arguments[0].number-loc2.arguments[0].number),Number(loc1.arguments[1].number-loc2.arguments[1].number)],True)
            node1.constraints.append(Function(name="constraint", arguments=[agent1,loc1,move1,time]))
            node2.constraints.append(Function(name="constraint", arguments=[agent2,loc2,move2,time]))

        node1.low_level(agent1.number,max_horizon)
        node2.low_level(agent2.number,max_horizon)

        return node1, node2

class CBS_Solver:

    def __init__(self, instance_file : str, greedy : bool = False, log_level : int = logging.INFO):
        self.instance_file = instance_file
        self.greedy = greedy
        self.solution = Solution()
        logger.setLevel(log_level)

    def preprocessing(self) -> None:

        def preprocessing_parser(model : Model, solution : Solution) -> bool:

            for atom in model.symbols(atoms=True):
                if(atom.name == 'init'):
                    solution.inits.append(atom)
                elif(atom.name == 'numOfRobots'):
                    solution.agents = list(range(1,atom.arguments[0].number+1))
                elif(atom.name == 'numOfNodes'):
                    solution.max_horizon = atom.arguments[0].number
                else:
                    solution.instance_atoms.append(atom)

            return False

        logger.debug("Preprocessing invoked")

        ctl : Control = Control(["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(PREPROCESSING_FILE)

        ctl.ground([("base",[])])

        ctl.solve(on_model=lambda model : preprocessing_parser(model,self.solution))


    def solve(self) -> Solution:

        open_queue : List[CTNode] = []
        solution_nodes : List[CTNode] = []
        root : CTNode
        current : CTNode

        try:

            logger.debug("Programm started")

            start_time : float = perf_counter()
            
            self.preprocessing()

            self.solution.max_horizon *=2 

            root = CTNode(None,None,self.solution.instance_atoms)

            logger.debug("Initializing first node")

            for agent in self.solution.agents:
                root.low_level(agent,self.solution.max_horizon)

            if(root.cost == inf):
                logger.info("No initial solution found!")
                exit()
                    
            open_queue.append(root)

            logger.debug("While loop started")

            while open_queue:

                current = open_queue.pop(0)

                first_conflict = current.validate_plans()

                if first_conflict:
                    node1, node2 = current.branch(first_conflict,self.solution.max_horizon)
                    if self.greedy:
                        if node1.cost < inf : open_queue.insert(0,node1)
                        if node2.cost < inf : open_queue.insert(0,node2)
                    else:
                        if node1.cost < inf : insort(open_queue,node1)
                        if node2.cost < inf : insort(open_queue,node2)
                        
                else:
                    solution_nodes.append(current)
                    break

        except KeyboardInterrupt:
            logger.info("Search terminated by keyboard interrupt")

        self.solution.execution_time = perf_counter() - start_time

        if solution_nodes:

            best_solution : CTNode = max(solution_nodes)

            self.solution.plans = best_solution.plans

            self.solution.cost = best_solution.cost

            self.solution.satisfied = True

            logger.info("Solution found")
            logger.info(f'Total model cost : {best_solution.cost}')

        else:
            logger.info("No solution found")

        return self.solution
    

if __name__ == '__main__':

    """Command line argument parsing"""
    parser : ArgumentParser = ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("-b", "--benchmark", default=False, action="store_true")
    parser.add_argument("-g", "--greedy", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
    args : Namespace = parser.parse_args()

    solution =  CBS_Solver(args.instance,args.greedy,logging.DEBUG if args.debug else logging.INFO).solve()

    if solution.satisfied:
        solution.save('plan.lp')

    if args.benchmark:
        logger.info(f"Execution time: {solution.execution_time:.3f}s")

    

    

        

