from __future__ import annotations
from bisect import insort
from typing import Dict, List, Tuple, Union
from unicodedata import name
from clingo import Control, Number, Function, Symbol, Model
from clingo.solving import SolveResult, SolveHandle
from time import perf_counter
from os import path
from argparse import ArgumentParser, Namespace
from sys import exit, stdout
from math import inf
import logging
from copy import deepcopy, copy
from solution import Solution
from math import ceil

"""Logging setup"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(message)s')
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')
PREPROCESSING_FILE : str = path.join(ENCODING_DIR,'setup.lp')
SAPF_FILE : str = path.join(ENCODING_DIR,'singleAgentPF_inc.lp')
MAPF_FILE : str = path.join(ENCODING_DIR,'multiAgentPF_inc.lp')
VALIADTION_FILE : str = path.join(ENCODING_DIR,'validate.lp')

META_AGENT_THRESHOLD : int = 2

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

    def __init__(self,plans : Dict[int,Dict[str,Union[List[Symbol],List[Symbol],int]]]=None, constraints : List[Symbol]=None, conflict_matrix : ConflictMatrix = None, atoms : List[Symbol]=None, cost : int = 0) -> None:
        self.plans = plans if plans else {}
        self.constraints = constraints if constraints else []
        self.conflic_matrix = conflict_matrix
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
        if not self.conflic_matrix:
            return self.low_level_sa(agent,horizon)
        elif self.conflic_matrix.is_meta_agent(agent):
            return self.low_level_ma(agent,horizon)
        else:
            return self.low_level_sa(agent,horizon)

    def low_level_ma(self,agent : int, horizon : int) -> bool:
        meta_agent : Tuple[int] = self.conflic_matrix.meta_agents[agent-1]
        ctl : Control = Control(['-Wnone'])
        step : int = 0 
        cost : int = 0
        ret : SolveResult = None
        handle : SolveHandle = None
        optimal_model : Model = None

        logger.debug(f"low level search invoked for meta agent {meta_agent}")

        for agt in meta_agent:
            if agt in self.plans:
                self.cost -= self.plans[agt]['cost']
            self.plans[agt] = {'positions' : [], 'occurs' : [], 'cost' : 0}

        ctl.load(MAPF_FILE)

        with ctl.backend() as backend:
            for atom in self.atoms + self.constraints:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

            for agt in meta_agent:
                fact = backend.add_atom(Function(name='planning',arguments=[Number(agt)]))
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
            with ctl.solve(yield_=True) as handle:
                ret = handle.get()
                step += 1
                if ret.satisfiable:
                    optimal_model = handle.model()
                    for optimal_model in handle:
                        pass
                    for atom in optimal_model.symbols(shown=True):
                        if atom.name == 'occurs':
                            self.plans[atom.arguments[0].arguments[1].number]['occurs'].append(atom)                          
                        else:
                            self.plans[atom.arguments[0].number]['positions'].append(atom)
                            if atom.name == 'goalReached':
                                cost += atom.arguments[1].number
                                self.plans[atom.arguments[0].number]['cost'] = atom.arguments[1].number

        self.cost += cost

        logger.debug(f"low level search terminated for meta agent {meta_agent} with joint cost {cost}")
                        
        return ret.satisfiable
        

    def low_level_sa(self,agent : int, horizon : int) -> bool:

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
            for plan in self.plans.values():
                for atom in plan['positions']:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])
            for atom in self.atoms:
                if atom.name == 'goal':
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])

        ctl.ground([('base',[])])

        ctl.solve(on_model=lambda model : validate_parser(model,conflicts))

        return conflicts[0] if conflicts else False

    def branch(self,conflict : Symbol, max_horizon : int) -> Tuple[CTNode,CTNode]:

        logger.debug("branch invoked")

        node1 : CTNode = CTNode(deepcopy(self.plans),copy(self.constraints),deepcopy(self.conflic_matrix),self.atoms,self.cost)
        node2 : CTNode = CTNode(deepcopy(self.plans),copy(self.constraints),deepcopy(self.conflic_matrix),self.atoms,self.cost)

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

class ConflictMatrix:

    def __init__(self, agents : List[Union[int,Tuple[int,...]]]) -> None:
        self.meta_agents = agents.copy()
        self.conflict_matrix : List[int] = [0] * int(len(self.meta_agents)*(len(self.meta_agents)-1) / 2)

    def is_meta_agent(self,agent : int) -> bool:
        return type(self.meta_agents[agent-1]) == tuple

    def merge(self, agent1 : int, agent2 : int) -> None:

        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents):

            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp

            agent1_t : Tuple[int,...] = self.meta_agents[agent1-1] if self.is_meta_agent(agent1) else (agent1,)
            agent2_t : Tuple[int,...] = self.meta_agents[agent2-1] if self.is_meta_agent(agent2) else (agent2,)
            new_agent : Tuple[int,...] = agent1_t + agent2_t
            for agent in new_agent:
                self.meta_agents[agent-1] = new_agent
            

    def update(self, agent1 : int, agent2: int) -> None:
        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents) and agent1 != agent2:
            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp
            self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-1)] += 1

    def should_merge(self,agent1 : int, agent2 : int, cthreshold : int) -> bool:
        return self._get_c_count(agent1,agent2) >= cthreshold

    def _c_count(self,agent1 : int, agent2 : int) -> int:
        return self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-(agent1)*(agent1+1)//2)-1]

    def _get_c_count(self, agent1 : int,agent2 : int) -> int:
        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents) and agent1 != agent2:
            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp
            total_conflicts : int = 0
            self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-(agent1)*(agent1+1)//2)-1] += 1
            
            if self.is_meta_agent(agent1) and self.is_meta_agent(agent2):
                for a1 in self.meta_agents[agent1-1]:
                    for a2 in self.meta_agents[agent2-1]:
                        total_conflicts += self._c_count(a1,a2)
            
            if self.is_meta_agent(agent1):
                for agent in self.meta_agents[agent1-1]:
                    total_conflicts += self._c_count(agent,agent2)

            if self.is_meta_agent(agent2):
                for agent in self.meta_agents[agent2-1]:
                    total_conflicts += self._c_count(agent,agent1)
            
            else:
                total_conflicts += self._c_count(agent1,agent2)

            return total_conflicts
                      
        else: 
            return 0

        


class CBSSolver:

    def __init__(self, instance_file : str, greedy : bool = False,meta : bool = True, log_level : int = logging.INFO) -> None:
        self.instance_file = instance_file
        self.greedy = greedy
        self.meta = meta
        self.solution = Solution()
        self.meta_agents : List[Union[int,Tuple[int]]] = []
        self.conflict_matrix : List[List[int]] = []
        logger.setLevel(log_level)

    def preprocessing(self) -> None:

        def preprocessing_parser(model : Model, solution : Solution) -> bool:

            for atom in model.symbols(atoms=True):
                if(atom.name == 'init'):
                    solution.inits.append(atom)
                elif(atom.name == 'numOfRobots'):
                    solution.agents = list(range(1,atom.arguments[0].number+1))
                elif(atom.name == 'numOfNodes'):
                    solution.num_of_nodes = atom.arguments[0].number
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
        max_iter : int 
        root : CTNode
        current : CTNode

        try:

            logger.debug("Programm started")

            start_time : float = perf_counter()
            
            self.preprocessing()

            max_iter = self.solution.num_of_nodes * 2 

            root = CTNode(atoms=self.solution.instance_atoms)

            if self.meta:
                root.conflic_matrix = ConflictMatrix(self.solution.agents)

            logger.debug("Initializing first node")

            for agent in self.solution.agents:
                root.low_level(agent,max_iter)

            if(root.cost == inf):
                logger.info("No initial solution found!")
                return self.solution
                    
            open_queue.append(root)

            logger.debug("While loop started")

            while open_queue:

                current = open_queue.pop(0)

                first_conflict : Union[Symbol,bool] = current.validate_plans()

                if first_conflict:

                    branch : bool = True

                    if self.meta:
                        agent1 : int = first_conflict.arguments[1].number
                        agent2 : int = first_conflict.arguments[2].number

                        if current.conflic_matrix.should_merge(agent1,agent2,META_AGENT_THRESHOLD):
                            logger.debug(f"Merged Agents {current.conflic_matrix.meta_agents[agent1-1]} and {current.conflic_matrix.meta_agents[agent2-1]}")
                            branch = False
                            current.conflic_matrix.merge(agent1,agent2)
                            current.low_level(agent1, max_iter)
                            if self.greedy:
                                if current.cost < inf : open_queue.insert(0,current)
                            else:
                                if current.cost < inf : insort(open_queue,current)

                    if branch:
                        node1, node2 = current.branch(first_conflict,max_iter)
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
    args : Namespace = parser.parse_args()

    solution =  CBSSolver(args.instance,args.greedy,logging.DEBUG if args.debug else logging.INFO).solve()

    if solution.satisfied:
        solution.save('plan.lp')

    if args.benchmark:
        logger.info(f"Execution time: {solution.execution_time:.3f}s")

    

    

        

