"""Imports"""
from __future__ import annotations
from bisect import insort
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union
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

"""
This file implements the conflict based search algorithm (CBS) based on Sharon, Guni & Stern, Roni & Felner, Ariel & Sturtevant, Nathan. (2015).
Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence. 219. 40-66. 10.1016/j.artint.2014.11.006.
It includes the basic CBS algorithm, as well as meta agent CBS (MA-CBS) and the bypassing (BP) and improved CBS (ICBS) extensions to the algorithm.
A more detailed description can be found in the report directory in this repository. 
"""

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

class Cost(Enum):
    SOC = 1
    MAKESPAN = 2
    GREEDY_SOC = 3
    GREEDY_MAKESPAN = 4

class CTNode:
    """
    CTNode represents a node in the constraint tree of the conflict based search algorithm.

    Attributes:
        plans : Dict[int,Plan]
            Stores current plans for all agents.
        conflic_matrix : ConflictMatrix
            Stores conflict count between all agents and current meta agents (MA-CBS only, else None).
        atoms : List[Symbol]
            Stores preprocessing atoms describing the current instance file.
        cost : int
            Stores sum of costs of all current plans.
        makespan : int
            Stores maximum makespan of all current plans.
        cost_function : Cost
            Stores a cost enum value representative of the cost function.
        constraint_count : int
            Stores number of constraints in all current plans.
        conflict_count : int
            Stores nubmer of conflicts between all current plans.
        conflicts : List[Symbol]  
            Stores conflic atoms between all current plans (ICBS).
        comp_value : Callable[...,int]
            Returns a value used to compare the node to other nodes, dependent on the chosen cost function.

    Methods:
        __init__(self,plans : Dict[int,Plan]=None, conflict_matrix : ConflictMatrix = None, atoms : List[Symbol]=None, cost : int = 0, constraint_count : int = 0) -> None
        __gt__(self, other : CTNode) -> bool
        __ge__(self, other : CTNode) -> bool
        __lt__(self, other : CTNode) -> bool
        __le__(self, other : CTNode) -> bool
        low_level(self,agent : int) -> bool
            Wrapper for low level search.
        low_level_ma(self,agent : int, horizon : int) -> bool
            Computes paths for all agents in meta agent.
        low_level_sa(self,agent : int, horizon : int) -> bool
            Computes path for agent.
        validate_plans(self) -> bool
            Determines conflicts between all current paths, returns True if no conflict are found else False.
        branch(self,conflict : Symbol, max_horizon : int) -> Union[Tuple[CTNode,CTNode],Tuple[CTNode,None]]
            Branches the CTNode along a conflict into new CTNodes.  
        clear_constraints(self, agent : int) -> int
            Deletes constraints in the plan of an agent.
    """

    def __init__(self,plans : Dict[int,Plan]=None, conflict_matrix : ConflictMatrix = None, atoms : List[Symbol]=None, cost : int = 0, makespan : int = 0, constraint_count : int = 0, cost_function : Cost = Cost.SOC) -> None:
        self.plans = plans if plans else {}
        self.conflic_matrix = conflict_matrix
        self.atoms = atoms
        self.cost = cost
        self.makespan = makespan
        self.cost_function : Cost = cost_function
        self.constraint_count = constraint_count
        self.conflict_count : int = 0
        self.conflicts : List[Symbol] = []
        self.comp_value : Callable[...,int]
        if self.cost_function == Cost.SOC:
            self.comp_value = lambda : self.cost
        elif self.cost_function == Cost.MAKESPAN:
            self.comp_value = lambda : self.makespan
        elif self.cost_function == Cost.GREEDY_SOC:
            self.comp_value = lambda : self.cost + self.conflict_count
        else: 
            self.comp_value = lambda : self.makespan + self.conflict_count
        
    def __gt__(self, other : CTNode) -> bool: 
        if self.comp_value() > other.comp_value():
            return True 
        elif self.comp_value() == other.comp_value():
            return self.constraint_count > other.constraint_count
        else:
            return False


    def __ge__(self, other : CTNode) -> bool:
        if self.comp_value() > other.comp_value():
            return True 
        elif self.comp_value() == other.comp_value():
            return self.constraint_count >= other.constraint_count
        else:
            return False


    def __lt__(self, other : CTNode) -> bool:
        return not self.__ge__(other)

    def __le__(self, other : CTNode) -> bool:
        return not self.__gt__(other)

    def low_level(self,agent : int, horizon : int) -> bool:
        """
        Wrapper function for low level search to allow for both
        regular and meta agent cbs.

        Args:
            agent : (meta) Agent for which a path should be computed.
            horizon : Upper bound on the path lenght for the incremental solving.

        Side effect: 
            self.cost : Updated with the change in cost of the new plan(s).
            self.plans : Updated with occurs and position atoms of the new plan(s)

        Returns:
            True, if a valid path was found with length <= horizon else False 
        """
        if not self.conflic_matrix:
            return self.low_level_sa(agent,horizon)
        elif self.conflic_matrix.is_meta_agent(agent):
            return self.low_level_ma(agent,horizon)
        else:
            return self.low_level_sa(agent,horizon)

    def low_level_ma(self,agent : int, horizon : int) -> bool:
        """
        Computes paths for all agents in the meta agent  

        Args:
            agent : Meta agent for which a path should be computed.
            horizon : Upper bound on the path lenght for the incremental solving.

        Side effect: 
            self.cost : Updated with the change in cost of the new plans.
            self.plans : Updated with occurs and position atoms of the new plans

        Returns:
            True if a valid paths where found with max length <= horizon else False 
        """

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
                    itr = iter(handle)
                    for optimal_model in itr:
                        pass
                    if optimal_model:
                        for atom in optimal_model.symbols(shown=True):
                            if atom.name == 'occurs':
                                self.plans[atom.arguments[0].arguments[1].number].occurs.append(atom)                          
                            else:
                                self.plans[atom.arguments[0].number].positions.append(atom)
                                if atom.name == 'goalReached' and atom.arguments[1]!=Supremum:
                                    cost += atom.arguments[1].number
                                    self.plans[atom.arguments[0].number].cost = atom.arguments[1].number
                                    self.makespan = max(self.makespan,atom.arguments[1].number)
                    else:
                        cost = inf
                        self.makespan = inf

        self.cost += cost

        logger.debug(f"low level search terminated for meta agent {meta_agent} with cost {cost}")
                        
        return ret.satisfiable
        

    def low_level_sa(self,agent : int, horizon : int) -> bool:
        """
        Computes path for agent  

        Args:
            agent : Agent for which a path should be computed.
            horizon : Upper bound on the path lenght for the incremental solving.

        Side effect: 
            self.cost : Updated with the change in cost of the new plan.
            self.plans : Updated with occurs and position atoms of the new plan

        Returns:
            True if a valid path was found with max length <= horizon else False 
        """

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

        plan.clear()

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

        self.makespan = max(self.makespan,cost)

        logger.debug(f"low level search terminated for agent {agent} with cost {cost}")

        self.cost += cost

        return plan.cost < inf

    def validate_plans(self) -> bool:
        """
        Determines conflicts between current plans 

        Side effect:
            self.conflict_count : Updated with the number of found conflicts
            self.conflicts : Updated with the atoms of all found conflicts
        
        Returns:
            True if no conflicts are found, else False

        """

        ctl : Control
        conflicts : List[Symbol] = []

        def validate_parser( model : Model, conflicts : List[Symbol]) -> bool:
            for atom in model.symbols(shown=True):
                conflicts.append(atom)
            return bool

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

        self.conflict_count = len(conflicts)

        self.conflicts = conflicts

        return not conflicts

    def branch(self,conflict : Symbol, horizon : int, copy : bool = False) -> Union[Tuple[CTNode,CTNode],Tuple[CTNode,None]]:
        """
        Branches the CTNode along a conflict into two new CTNodes, each with an additional constraint for one of the conflicting agents.
        Additionally implements bypassing, which will return only the Node containing the bypass if a valid bypass is found.

        Args:
            conflict : Conflict atom of the conflict that is supposed to be branched.
            horizon : Upper bound on the path lenght for the low level search.
            copy : If True, both nodes will be new nodes insted of reusing the existing node.

        Side effects:
            plans : Updates plan of conflicting agent with path under new constraint.
            cost : Updates cost with the cost of the new plan.
            constraint_count :  Increments constraint_count by one.
.
        Returns:
            New CTNodes, replanned with additional constraints or only one Node of a valid bypass is found. 
        """


        logger.debug("branch invoked")
        conflict_type : str = conflict.arguments[0].name
        agent1 : Number = conflict.arguments[1]
        agent2 : Number = conflict.arguments[2]
        constraint1_type : str = 'meta_constraint' if self.conflic_matrix and self.conflic_matrix.is_meta_agent(agent1.number) else 'constraint'
        constraint2_type : str = 'meta_constraint' if self.conflic_matrix and self.conflic_matrix.is_meta_agent(agent2.number) else 'constraint'
        time : Number = conflict.arguments[4]
        old_cost : int = self.cost
        old_makespan : int = self.makespan
        old_ccount : int = self.conflict_count
        old_plan : Plan = deepcopy(self.plans[agent1.number]) if not copy else None
        constraint1 : Symbol
        constraint2 : Symbol
        node1 : CTNode
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

        if copy:
            node1 = CTNode(deepcopy(self.plans),deepcopy(self.conflic_matrix),self.atoms,old_cost,old_makespan,old_ccount + 1,self.cost_function)
        else:
            node1 = self 
            self.constraint_count += 1
        node1.plans[agent1.number].constraints.append(constraint1)
        node1.low_level(agent1.number,horizon)
        node1.validate_plans()
        if node1.cost <= old_cost and node1.conflict_count < old_ccount:
            return node1, None
        
        node2 = CTNode(deepcopy(self.plans),deepcopy(self.conflic_matrix),self.atoms,old_cost,old_makespan,old_ccount + 1,self.cost_function)
        node2.plans[agent2.number].constraints.append(constraint2)
        if not copy:
            node2.plans[agent1.number] = old_plan
        node2.low_level(agent2.number,horizon)
        node2.validate_plans()

        if node2.cost <= old_cost and node2.conflict_count < old_ccount:
            return node2, None  
        else: return node1, node2

    def icbs_branch(self, horizon : int) -> Union[Tuple[CTNode,CTNode],Tuple[CTNode,None]]:
        """
        Branching funtion for ICBS. Examines every conflict in the current node and determines its cardinality.
        Will branch one conflict with priority order cardinal > semi-cardinal > bypass > non-cardinal.
        If a cardinal conflict is found, its minmal cost is returned as a lower bound. 

        Args:
            horizon : Upper bound for low level path finding

        Returns:
            The two CTNodes of the chosen conflict. If the conflict is cardinal the minimal cost is
            returned as a third return value, else None. If a valid bypass is returned, the second node is None.
        """
        bypass : Union[CTNode,None] = None
        ncnode1 : Union[CTNode,None] = None
        ncnode2 : Union[CTNode,None] = None
        scnode1 : Union[CTNode,None] = None
        scnode2 : Union[CTNode,None] = None
        node1 : CTNode
        node2 : CTNode

        logger.debug("ICBS branch invoked")

        for conflict in self.conflicts:
            node1, node2 = self.branch(conflict,horizon,True)
            if node2:
                if (node1 > self and node2 > self) or node1.conflict_count == 0 or node2.conflict_count == 0:
                    return node1, node2
                elif node1 > self or node2 > self:
                    scnode1, scnode2 = node1, node2
                else:
                    ncnode1, ncnode2 = node1, node2
            else:
                bypass = node1
            
            if scnode1:
                return scnode1, scnode2
            elif bypass:
                return bypass, None
            else:
                return ncnode1, ncnode2

    def clear_constraints(self, agent : int) -> int:
        """
            Removes all constraints from an agents plan 

            Args:
                agent : Agent whos constraints should be cleared.

            Side effects:
                constraint_count : Decremented by number of removed constraints.

            Returns:
                Number of removed constraints
        """
        count : int = 0
        plan : Plan
        if self.conflic_matrix and self.conflic_matrix.is_meta_agent(agent):
            for ag in self.conflic_matrix.meta_agents[agent-1]:
                plan = self.plans[ag]
                count += len(plan.constraints)
                plan.constraints = []

        else:
            plan = self.plans[agent]
            count += len(plan.constraints)
            plan.constraints = []
        self.constraint_count -= count
        return count

class ConflictMatrix:
    """
    ConflictMatrix represents the conflict matrix used for MA-CBS. Keeps track of the number of conflicts
    between all agents and the current meta agents.

    Attributes:
        meta_agents : List[Union[int,Tuple[int,...]]]
            Stores a reference to the respective meta agent for all agents.
        conflict_matrix : List[int]
            Stores the number of conflicts between all pairs of agents.

    Methods:
        __init__(self, agents : List[Union[int,Tuple[int,...]]]) -> None
        is_meta_agent(self,agent : int) -> bool
            Returns True if agents is part of a meta agent, else False.
        merge(self, agent1 : int, agent2 : int) -> None
            Merges two (meta) agents to a joint meta agent.
        update(self, agent1 : int, agent2: int) -> None
            Increments the conflict matrix for two agents by one.
        should_merge(self,agent1 : int, agent2 : int, cthreshold : int) -> bool
            Returns True if the number of conflicts between two agents exeeds the threshold, else False.
        _c_count(self,agent1 : int, agent2 : int) -> int
            (Should not be called directly). Returns conflict matrix entry for a pair of agents.
        _get_c_count(self, agent1 : int,agent2 : int) -> int
            (Should not be called directly). Returns the number of conflicts between two (meta) agents.
        clear_cmatrix(self) -> None    
            Resets conflict count to zero for all agent pairs.
    """


    def __init__(self, agents : List[Union[int,Tuple[int,...]]]) -> None:
        self.meta_agents = agents.copy()
        self.conflict_matrix : List[int] = [0] * int(len(self.meta_agents)*(len(self.meta_agents)-1) / 2)

    def is_meta_agent(self,agent : int) -> bool:
        """
        Checks if agent is part of a meta agent.

        Args:
            agent : Agent to be checked.

        Returns:
            True if agent is part of meta agent, else False
        """
        return type(self.meta_agents[agent-1]) == tuple

    def merge(self, agent1 : int, agent2 : int) -> None:
        """
        Merges two (meta) agents to a joint meta agent. 

        Args:
            agent1 : Agent to be merged (If agent is part of meta agent the whole meta agent will be merged). 
            agent2 : Agent to be merged (If agent is part of meta agent the whole meta agent will be merged).

        Side effect:
            meta_agents : Entries for all merged agents are updated with reference to new meta agent.
        """

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
        """
        Updates conflict count for a pair of agents.

        Args:
            agent1 : First agent in conflict.
            agent2 : Second agent in conflict.

        Side effect:
            conflict_matrix : Entry for (agent1,agent2) is incremented by one.
        """
        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents) and agent1 != agent2:
            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp
            self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-1)] += 1

    def should_merge(self,agent1 : int, agent2 : int, cthreshold : int) -> bool:
        """
        Checks if two agents should be merged according to threshold.

        Args:
            agent1 : First agent to be checked.
            agent1 : Second agent to be checked.
            cthreshold : Threshold for number of conflicts at which two agents are beeing merged.

        Returns:
            True if number of conflicts between agents has reached threshold, else False
        """

        if 0 < agent1 <= len(self.meta_agents) and 0 < agent2 <= len(self.meta_agents) and self.meta_agents[agent1-1] != self.meta_agents[agent2-1]:
            if agent1 > agent2:
                temp : int = agent1
                agent1 = agent2
                agent2 = temp
            return self._get_c_count(agent1,agent2) >= cthreshold
        else:
            return False

    def _c_count(self,agent1 : int, agent2 : int) -> int:
        """
        Should not be called directly. Gets entry in conflict matrix.

        Args:
            agent1 : First agent in entry.
            agent2 : Second agent in entry.

        Returns:
            Entry in conflict matrix for (agent1,agent2). 
        """

        return self.conflict_matrix[(agent1-1)*len(self.meta_agents)+(agent2-(agent1)*(agent1+1)//2)-1]

    def _get_c_count(self, agent1 : int,agent2 : int) -> int:
        """
        Should not be called directly. Gets number of conflicts between two (meta)agents.

        Args:
            agent1 : First agent.
            agent2 : Second agent.

        Returns:
            Returns number of all conflicts between the (meta)agents. 
        """
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
        
    def clear_cmatrix(self) -> None:
        """
        Resets all conflict counts in conflict matrix to zero.

        Side effect:
            conflict_matrix : Updated to zero for all agent pairs.
        """
        self.conflict_matrix : List[int] = [0] * len(self.conflict_matrix)


class CBSSolver:
    """
    Implements solver object which executes the CBS algorithm with given instance file and parameters

    Attributes:
        instance_file : str
            Path of the asprilo instance file that is to be solved.
        cost_function : Cost
            Stores cost enum value representing the chosen cost function
        meta : bool
            Activates MA-CBS if True.
        icbs : bool
            Activates ICBS if True.
        threshold :
            Sets conflict treshold for (meta)agent merging in MA-CBS.
        solution : Solution
            Contains the solution found by CBS.
        timeout : int
            Maximum number of seconds that the solver is allowed to run

    Methods:
        preprocessing(self) -> None
            Parses the instance file and initializes the solver.
        solve(self) -> Solution
            Executes main CBS algorithm
    """

    def __init__(self, instance_file : str, greedy : bool = False, makespan : bool = False, meta : bool = False, icbs : bool = False, threshold : int = 2, log_level : int = logging.INFO,timeout : int = inf) -> None:
        self.instance_file = instance_file
        self.meta = meta
        self.icbs = icbs
        self.meta_threshold = threshold
        self.timeout = timeout
        self.cost_function : Cost = Cost.SOC
        self.solution = Solution()
        logger.setLevel(log_level)
        if makespan:
            if greedy:
                self.cost_function = Cost.GREEDY_MAKESPAN
            else:
                self.cost_function = Cost.MAKESPAN
        elif greedy:
            self.cost_function = Cost.GREEDY_SOC

    def preprocessing(self) -> None:
        """
        Parses the instance file and initalized solver with information about the instance.

        Side effects:
            solution.inits : Updated with init atoms of instance file for the resulting plan.
            solution.agents : Updated with list contaning all agents in the instance file.
            solution.num_of_nodes : Updated with number of nodes of the instance.
            solution.plans : Plans for all agents are initialized with their goal.
            solution.instance_atoms : Updated with atoms, describing the instance.
        """

        def preprocessing_parser(model : Model) -> bool:
            """Parser function for the model of the preprocessing aps file."""

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
        """
        Executes the main CBS algorithm and extensions, depending on the set options. 

        Side effect:
            solution : Updated with plans and cost of the found solution and execution time

        Returns:
            Solution Object with the solution obtained by CBS.
        """

        open_queue : List[CTNode] = []
        solution_nodes : List[CTNode] = []
        max_iter : int 
        root : CTNode
        current : CTNode
        node1 : CTNode
        node2 : CTNode

        try:

            logger.debug("Programm started")

            start_time : float = perf_counter()
            
            self.preprocessing()

            max_iter = self.solution.num_of_nodes * 2 

            root = CTNode(atoms=self.solution.instance_atoms,plans=self.solution.plans,cost_function=self.cost_function)

            if self.meta:
                root.conflic_matrix = ConflictMatrix(self.solution.agents)

            logger.debug("Initializing first node")

            for agent in self.solution.agents:
                root.low_level(agent,max_iter)

            if root.cost == inf:
                logger.info("No initial solution found!")
                return self.solution

            if root.validate_plans():
                solution_nodes.append(root)
                    
            open_queue.append(root)

            logger.debug("While loop started")

            while open_queue and perf_counter() - start_time < self.timeout:


                if not solution_nodes:

                    current = open_queue.pop(0)

                    branch : bool = True

                    if self.meta:
                        agent1 : int = current.conflicts[0].arguments[1].number
                        agent2 : int = current.conflicts[0].arguments[2].number

                        if current.conflic_matrix.should_merge(agent1,agent2,self.meta_threshold):
                            logger.debug(f"Merged Agents {current.conflic_matrix.meta_agents[agent1-1]} and {current.conflic_matrix.meta_agents[agent2-1]}")
                            branch = False
                            current.conflic_matrix.merge(agent1,agent2)
                            if self.icbs:
                                current.conflic_matrix.clear_cmatrix()
                                for agent in set(current.conflic_matrix.meta_agents):
                                    ag : int = agent[0] if type(agent) == tuple else agent
                                    if current.clear_constraints(ag):
                                        current.low_level(ag,max_iter)
                                if current.validate_plans():
                                    solution_nodes.append(current)
                                    break
                                open_queue = [current]
                            else:
                                current.low_level(agent1, max_iter)
                                if current.cost < inf:
                                    if current.validate_plans():
                                        solution_nodes.append(current)
                                        break
                                    insort(open_queue,current)

                    if branch:
                        if self.icbs:
                            node1, node2 = current.icbs_branch(max_iter)
                        else:
                            node1, node2 = current.branch(current.conflicts[0],max_iter) 
                        if node1.conflict_count == 0:
                            solution_nodes.append(node1) 
                            break
                        elif node2 and node2.conflict_count == 0:
                            solution_nodes.append(node2) 
                            break
                        insort(open_queue,node1)
                        if node2: insort(open_queue,node2)
                        
                else:
                    break

        except KeyboardInterrupt:
            logger.info("Search terminated by keyboard interrupt")
            self.solution.plans = current.plans
            self.solution.cost = current.cost

        self.solution.execution_time = perf_counter() - start_time

        if solution_nodes:

            best_solution : CTNode = max(solution_nodes)

            self.solution.plans = best_solution.plans

            self.solution.get_soc()

            self.solution.get_makespan()

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

    

    

        

