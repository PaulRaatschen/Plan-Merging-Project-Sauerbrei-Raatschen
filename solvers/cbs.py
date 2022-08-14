from __future__ import annotations
from bisect import insort
from typing import Dict, List, Tuple, Union
from clingo import Control, Number, Function, Symbol, Model, Supremum
from clingo.solving import SolveResult, SolveHandle
from time import perf_counter
from os import path
from argparse import ArgumentParser, Namespace
from sys import stdout
from math import inf
import logging
from copy import deepcopy
from solution import Solution, Plan


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
SAPF_FILE : str = path.join(ENCODING_DIR,'single_agent_pf_cbs.lp')
MAPF_FILE : str = path.join(ENCODING_DIR,'multi_agent_pf.lp')
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

    def __init__(self,plans : Dict[int,Plan]=None, conflict_matrix : ConflictMatrix = None, atoms : List[Symbol]=None, cost : int = 0, constraint_count : int = 0) -> None:
        self.plans = plans if plans else {}
        self.conflic_matrix = conflict_matrix
        self.atoms = atoms
        self.cost = cost
        self.constraint_count = constraint_count
        self.c_count : int = 0
        self.conflict : Union[Symbol,None] = None

    def __gt__(self, other : CTNode) -> bool:
        if self.cost > other.cost:
            return True 
        elif self.cost == other.cost:
            return self.constraint_count > other.constraint_count
        else:
            return False

    def __ge__(self, other : CTNode) -> bool:
        if self.cost > other.cost:
            return True 
        elif self.cost == other.cost:
            return self.constraint_count >= other.constraint_count
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
            self.cost -= self.plans[agt].cost
            self.plans[agt].clear()

        ctl.load(MAPF_FILE)

        with ctl.backend() as backend:
            for atom in self.atoms:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

            for agt in meta_agent:
                for constraint in self.plans[agt].constraints + [Function(name='planning',arguments=[Number(agt)]), self.plans[agt].goal]:
                    fact = backend.add_atom(constraint)
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
                            self.plans[atom.arguments[0].arguments[1].number].occurs.append(atom)                          
                        else:
                            self.plans[atom.arguments[0].number].positions.append(atom)
                            if atom.name == 'goalReached' and atom.arguments[1]!=Supremum:
                                cost += atom.arguments[1].number
                                self.plans[atom.arguments[0].number].cost = atom.arguments[1].number

        self.cost += cost

        logger.debug(f"low level search terminated for meta agent {meta_agent} with cost {cost}")
                        
        return ret.satisfiable
        

    def low_level_sa(self,agent : int, horizon : int) -> bool:

        ctl : Control = Control(['-Wnone',f'-c r={agent}'])
        step : int = 0 
        ret : SolveResult = None
        plan : Plan = self.plans[agent]
        cost : int = 0

        def low_level_parser(model : Model, agent : int) -> bool:
            for atom in model.symbols(shown=True):
                if atom.name == 'occurs':
                    self.plans[agent].occurs.append(atom)
                else:
                    self.plans[agent].positions.append(atom)
        
        logger.debug(f"low level search invoked for agent {agent}")

        self.cost -= plan.cost

        plan.clear

        ctl.load(SAPF_FILE)

        with ctl.backend() as backend:
            for atom in self.atoms + plan.constraints + [plan.goal]:
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
                    ret, step = ctl.solve(on_model=lambda model : low_level_parser(model,agent)), step + 1
                    
        cost = inf if (ret and (not ret.satisfiable)) else step - 1

        self.plans[agent].cost = cost

        logger.debug(f"low level search terminated for agent {agent} with cost {cost}")

        self.cost += cost

        return plan.cost < inf

    def validate_plans(self) -> bool:

        ctl : Control
        conflicts : List[Symbol] = []

        def validate_parser( model : Model, conflicts : List[Symbol]) -> List[Symbol]:
            for atom in model.symbols(shown=True):
                conflicts.append(atom)
            return False

        logger.debug("Validate plans invoked")

        ctl = Control(['-Wnone'])

        ctl.load(VALIADTION_FILE)

        with ctl.backend() as backend:
            for plan in self.plans.values():
                fact = backend.add_atom(plan.goal)
                backend.add_rule([fact])
                for atom in plan.positions:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])
                    
        ctl.ground([('base',[])])

        ctl.solve(on_model=lambda model : validate_parser(model,conflicts))

        self.c_count = len(conflicts)

        self.conflict = conflicts[0] if conflicts else None

        return conflicts

    def branch(self,conflict : Symbol, max_horizon : int) -> Union[Tuple[CTNode,CTNode],Tuple[CTNode,None]]:

        logger.debug("branch invoked")

        conflict_type : str = conflict.arguments[0].name
        agent1 : Number = conflict.arguments[1]
        agent2 : Number = conflict.arguments[2]
        constraint1_type : str = 'meta_constraint' if self.conflic_matrix and self.conflic_matrix.is_meta_agent(agent1.number) else 'constraint'
        constraint2_type : str = 'meta_constraint' if self.conflic_matrix and self.conflic_matrix.is_meta_agent(agent2.number) else 'constraint'
        time : Number = conflict.arguments[4]
        old_cost : int = self.cost
        old_ccount : int = self.c_count
        old_plan : Plan = deepcopy(self.plans[agent1.number])
        constraint1 : Symbol
        constraint2 : Symbol
        node2 : CTNode

        if conflict_type == 'vertex':
            loc : Function = conflict.arguments[3]
            constraint1 = Function(name=constraint1_type, arguments=[agent1,loc,time])
            constraint2 = Function(name=constraint2_type, arguments=[agent2,loc,time])

        else:
            loc1 : Function = conflict.arguments[3].arguments[0]
            loc2 : Function = conflict.arguments[3].arguments[1]
            move1 : Function = Function('',[Number(loc2.arguments[0].number-loc1.arguments[0].number),Number(loc2.arguments[1].number-loc1.arguments[1].number)],True)
            move2 : Function = Function('',[Number(loc1.arguments[0].number-loc2.arguments[0].number),Number(loc1.arguments[1].number-loc2.arguments[1].number)],True)
            constraint1 = Function(name=constraint1_type, arguments=[agent1,loc1,move1,time])
            constraint2 = Function(name=constraint2_type, arguments=[agent2,loc2,move2,time])


        self.constraint_count += 1
        self.plans[agent1.number].constraints.append(constraint1)
        self.low_level(agent1.number,max_horizon)
        self.validate_plans()
        if self.cost <= old_cost and self.c_count < old_ccount:
            return self, None
        
        node2 = CTNode(deepcopy(self.plans),deepcopy(self.conflic_matrix),self.atoms,old_cost,self.constraint_count)
        node2.plans[agent2.number].constraints.append(constraint2)
        node2.plans[agent1.number] = old_plan
        node2.low_level(agent2.number,max_horizon)
        node2.validate_plans()

        if node2.cost <= old_cost and node2.c_count < old_ccount:
            return node2, None  
        else: return self, node2

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
        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents) and agent1 != agent2:
            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp
            return self._get_c_count(agent1,agent2) >= cthreshold
        else:
            return False

    def _c_count(self,agent1 : int, agent2 : int) -> int:
        return self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-(agent1)*(agent1+1)//2)-1]

    def _get_c_count(self, agent1 : int,agent2 : int) -> int:
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
        


class CBSSolver:

    def __init__(self, instance_file : str, greedy : bool = False,meta : bool = False, meta_threshold : int = 2, log_level : int = logging.INFO) -> None:
        self.instance_file = instance_file
        self.greedy = greedy
        self.meta = meta
        self.meta_threshold = meta_threshold
        self.solution = Solution()
        self.meta_agents : List[Union[int,Tuple[int]]] = []
        self.conflict_matrix : List[List[int]] = []
        logger.setLevel(log_level)

    def preprocessing(self) -> None:

        def preprocessing_parser(model : Model) -> bool:

            for atom in model.symbols(atoms=True):
                if atom.name == 'init':
                    self.solution.inits.append(atom)
                elif atom.name == 'numOfRobots':
                    self.solution.agents = list(range(1,atom.arguments[0].number+1))
                elif atom.name == 'numOfNodes':
                    self.solution.num_of_nodes = atom.arguments[0].number
                elif atom.name == 'goal':
                    self.solution.plans[atom.arguments[0].number] = Plan(goal=atom)
                else:
                    self.solution.instance_atoms.append(atom)

            return False

        logger.debug("Preprocessing invoked")

        ctl : Control = Control(["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(PREPROCESSING_FILE)

        ctl.ground([("base",[])])

        ctl.solve(on_model=preprocessing_parser)


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

            root = CTNode(atoms=self.solution.instance_atoms,plans=self.solution.plans)

            if self.meta:
                root.conflic_matrix = ConflictMatrix(self.solution.agents)

            logger.debug("Initializing first node")

            for agent in self.solution.agents:
                root.low_level(agent,max_iter)

            if root.cost == inf:
                logger.info("No initial solution found!")
                return self.solution

            if not root.validate_plans():
                solution_nodes.append(root)
                    
            open_queue.append(root)

            logger.debug("While loop started")

            while open_queue:

                if not solution_nodes:

                    current = open_queue.pop(0)

                    branch : bool = True

                    if self.meta:
                        agent1 : int = current.conflict.arguments[1].number
                        agent2 : int = current.conflict.arguments[2].number

                        if current.conflic_matrix.should_merge(agent1,agent2,self.meta_threshold):
                            logger.debug(f"Merged Agents {current.conflic_matrix.meta_agents[agent1-1]} and {current.conflic_matrix.meta_agents[agent2-1]}")
                            branch = False
                            current.conflic_matrix.merge(agent1,agent2)
                            current.low_level(agent1, max_iter)
                            if current.cost < inf:
                                if not current.validate_plans():
                                    solution_nodes.append(current)
                                    break
                                if self.greedy:
                                    open_queue.insert(0,current)
                                else:
                                    insort(open_queue,current)

                    if branch:
                        node1, node2 = current.branch(current.conflict,max_iter)
                        if node1.c_count == 0:
                            solution_nodes.append(node1) 
                            break
                        elif node2 and node2.c_count == 0:
                            solution_nodes.append(node2) 
                            break

                        if self.greedy:
                            open_queue.insert(0,node1)
                            if node2: open_queue.insert(0,node2)
                        else:
                            insort(open_queue,node1)
                            if node2: insort(open_queue,node2)
                        
                else:
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
    parser.add_argument("-m", "--meta", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--treshold",default=2,type=int)
    args : Namespace = parser.parse_args()

    solution =  CBSSolver(args.instance,args.greedy,args.meta, args.threshold,logging.DEBUG if args.debug else logging.INFO).solve()

    if solution.satisfied:
        solution.save('plan.lp')

    if args.benchmark:
        logger.info(f"Execution time: {solution.execution_time:.3f}s")

    

    

        

