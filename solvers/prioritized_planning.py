"""Imports"""
from __future__ import annotations
from typing import Callable, List, Union
from clingo import Control, Number, Function, Model
from time import perf_counter
from math import inf
from os import path
from argparse import ArgumentParser, Namespace
from sys import stdout
from solution import Solution, Plan
import permutation_tools as pt
import logging

"""
This file implements a prioritized planning algorithm for asprilo instances. A more detailed description can be found in the report directory 
of this repository.
"""

"""Logging setup"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(message)s')
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)



"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')
PREPROCESSING_FILE : str = path.join(ENCODING_DIR,'setup.lp')
SAPF_FILE : str = path.join(ENCODING_DIR,'single_agent_pf.lp')
CONFLICT_DETECTION_FILE : str = path.join(ENCODING_DIR,'validate.lp')
SAPF_NC_FILE : str = path.join(ENCODING_DIR,'single_agent_pf_nc.lp')

class PrioritizedPlanningSolver:
    """
    Implements a solver object which executes the prioritized planning algorithm.

    Attributes:
        instance_file : str
            Path to the asprilo instance file that is to ve solved.
        optimize : bool 
            If True, changes the priotization order of the agents to better avoid conflicts (will increase runtime).
        backtrack : bool
            If True, allows for backtrakking i.e. changes in agent order if the initial ordering does not lead to a solution.
        max_depth : int
            Determines the maximum allowed number of ordering changes for backtrakking.
        max_horizon : int
            Maximum path lenght for individual agent pathfinding.
        solution : Solution
            Stores the solution obtained by prioritized planning.
        

    """

    def __init__(self,instance_file : str, optimize : bool = False, backtrack : bool = False, maxdepth : int = 10, log_level : int = logging.INFO) -> None:
        self.instance_file = instance_file
        self.optimize = optimize
        self.backtrack = backtrack
        self.max_depth = maxdepth
        self.max_horion : int = 0
        self.solution : Solution = Solution()
        logger.setLevel(log_level)


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
            """Parse function for model created by preprocessing asp file"""

            for atom in model.symbols(atoms=True):
                if atom.name == 'init':
                    self.solution.inits.append(atom)
                elif atom.name == 'numOfRobots':
                    self.solution.agents = list(range(1,atom.arguments[0].number+1))
                elif atom.name == 'numOfNodes':
                    self.solution.num_of_nodes = atom.arguments[0].number
                elif atom.name == 'goal':
                    if atom.arguments[0].number in self.solution.plans:
                        self.solution.plans[atom.arguments[0].number].goal = atom
                    else: self.solution.plans[atom.arguments[0].number] = Plan(goal=atom)
                elif atom.name == 'position':
                    if atom.arguments[0].number in self.solution.plans:
                        self.solution.plans[atom.arguments[0].number].initial = atom
                    else: self.solution.plans[atom.arguments[0].number] = Plan(initial=atom)
                else:
                    self.solution.instance_atoms.append(atom)

            return False

        logger.debug("Preprocessing invoked")

        ctl : Control = Control(["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(PREPROCESSING_FILE)

        ctl.ground([("base",[])])

        ctl.solve(on_model=preprocessing_parser)


    def optimize_schedule(self) -> List[int]:
        """
        Changes the prioritization order of agents according to the number of conflicts
        between the shortest path of an agent with the shortest paths of the other agents.
        (Will increase runtime and a better solution is not guaranteed).

        Returns:
            New ordering of agents.
        """

        logger.debug("Optimization invoked")

        ctl : Control
        schedule : List[int] = []

        def optimization_parser(model : Model, schedule : List[int]) -> bool:
            """Parser function for model created by no conflict detection pathfinding asp file"""

            agent_conflicts : List[List[int]] = [[agent,0,self.solution.initial_plans[agent].cost] for agent in self.solution.agents]

            for atom in model.symbols(shown=True):
                agent_conflicts[atom.arguments[1].number-1][1] += 1

            schedule.extend([x[0] for x in sorted(agent_conflicts,key=lambda x : (x[1],x[2]))])

            return False

        self.solution.get_initial_plans()        

        ctl = Control(arguments=['-Wnone'])

        ctl.load(CONFLICT_DETECTION_FILE)

        with ctl.backend() as backend:
            for plan in self.solution.initial_plans.values():
                for position in plan.positions:
                    fact = backend.add_atom(position)
                    backend.add_rule([fact])

        ctl.ground([("base",[])])

        ctl.solve(on_model=lambda model : optimization_parser(model,schedule))

        logger.debug(f'New Schedule: {schedule}')

        return schedule if schedule else self.solution.agents.copy()

    def plan_path(self, agent : int) -> bool:
        """
        Computes the shortest path for an agent, avoiding collisions with allready planned paths.

        Args:
            agent : Agent for which a path should be planned

        Side effect: 
            self.solutions.plans : Updates the plan for the planning agent.

        Returns:
            True if a valid paths was found, else False.
        """

        ctl : Control
        cost : int = 0
        max_iter : int = self.solution.num_of_nodes * 2
  
        def plan_path_parser(model : Model, agent : int) -> bool:
            for atom in model.symbols(shown=True):
                    if atom.name == 'occurs':
                        self.solution.plans[agent].occurs.append(atom)
                    else:
                        self.solution.plans[agent].positions.append(atom)
            return False

        logger.debug(f'Planning for agent {agent}')

        self.solution.clear_plan(agent)

        ctl = Control(arguments=['-Wnone',f'-c r={agent}'])

        ctl.load(SAPF_FILE)

        with ctl.backend() as backend:
            fact = backend.add_atom(self.solution.plans[agent].initial)
            backend.add_rule([fact])

            for atom in self.solution.instance_atoms:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

            for plan in self.solution.plans.values():
                for atom in plan.positions + [plan.goal]:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])

        cost = self.incremental_solving(ctl,max_iter,lambda model : plan_path_parser(model,agent))

        self.solution.plans[agent].cost = cost

        logger.debug(f'Planning finished with cost {cost}')

        return cost < inf    

    def solve(self) -> Solution:
        """
        Executes the prioritized planning algorithm.

        Side effect: 
            Updates plans in solution and execution time. 

        Returns:
            Solution object with the solution obtained by prioritized planning.
        """

        logger.debug("Programm started")
        
        self.preprocessing()

        ordering : List[int] = self.solution.agents.copy()
        orderings : List[List[int]] = []
        finished_index : int = 0 
        t_start : float = perf_counter()
        satisfied : bool
        perm_index : Union[None,int]
        agent_to_swap : int
        num_of_agents : int = len(ordering)

        if self.optimize:
            ordering = self.optimize_schedule()
            

        
        for _ in range(self.max_depth):

            satisfied = True

            for index, agent in  enumerate(ordering[finished_index:]):

                if not self.plan_path(agent):
                    satisfied = False
                    logger.debug(f'No solution found for agent {agent}')

                    if self.backtrack:
                        finished_index = finished_index + index - 1
                        if finished_index < 0:
                            logger.debug(f'No solution possible')
                            self.backtrack = False
                            break
                        pt.update_pos(pt.partial_perm_index(ordering[:finished_index+1],num_of_agents),orderings,num_of_agents)
                        agent_to_swap = ordering[finished_index]
                        ordering[finished_index] = ordering[finished_index+1]
                        ordering[finished_index+1] = agent_to_swap
                        
                        perm_index = pt.update_pos([pt.permutation_index(ordering)],orderings,num_of_agents)

                        if perm_index:
                            if perm_index < 0:
                                logger.debug("Exhauset all orderings")
                                self.backtrack = False
                            else:
                                finished_index = 0
                                old_order : List[int] = ordering
                                ordering = pt.index_to_perm(perm_index,num_of_agents)
                                while old_order[finished_index] == ordering[finished_index] and finished_index < num_of_agents:
                                    finished_index += 1
                                for ag in ordering[finished_index+1:]:
                                    self.solution.clear_plan(ag)
                        else:
                            self.solution.clear_plan(agent_to_swap)
                        logger.debug(f'New ordering : {ordering}')
                        
                        break

            if not self.backtrack or satisfied:
                break

        logger.info("Planning finished")

        self.solution.execution_time = perf_counter() - t_start

        if satisfied:
            self.solution.satisfied = True
            logger.info("Global solution found")
        else:
            logger.info("No global solution found")
        
        return self.solution

    @staticmethod   
    def incremental_solving(ctl : Control, max_horizon : int, model_parser : Callable[[Model],bool]) -> int:
        """
        Helper method for the multishot asp pathfinding

        Args:
            ctl : Clingo control object with preloaded facts and rules.
            max_horizon : Upper bound for the amount of clingo calls i.e lenght of the path.
            model_parser : Function passed to the Control.solve() call to parse the resulting model.

        Returns:
            Number of clingo calls i.e. pathlength if a path was found or max_horizon + 1 if not.
        """

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



if __name__ == "__main__":

    """Command line argument parsing"""
    parser : ArgumentParser = ArgumentParser()
    parser.add_argument("instance", type=str)
    parser.add_argument("-b", "--benchmark", default=False, action="store_true",help="Outputs execution time and solution statistics.")
    parser.add_argument("-o", "--optimize", default=False, action="store_true",help="Enables initial agent schedule optimization.")
    parser.add_argument("--backtrack",default=False, action="store_true",help="Enables backtracking if the current ordering does not lead to a solution.")
    parser.add_argument("--maxdepth",default=10,type=int,help="Set the maximum amount of schedule changes for backtracking.")
    parser.add_argument("--debug", default=False, action="store_true",help="Makes solving process verbose for debugging purposes.")
    args : Namespace = parser.parse_args()

    solution = PrioritizedPlanningSolver(args.instance,args.optimize,args.backtrack,args.maxdepth,logging.DEBUG if args.debug else logging.INFO).solve()

    solution.save('plan.lp')

    if args.benchmark:
            logger.info(f'Execution time : {solution.execution_time:.2f}s')
            logger.info(f'Sum of costs : {solution.get_soc()}')
            logger.info(f'Sum of costs : {solution.get_makespan()}')
            logger.info(f'Total moves : {solution.get_total_moves()}')
    