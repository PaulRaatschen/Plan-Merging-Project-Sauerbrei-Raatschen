from copy import deepcopy
from msvcrt import SEM_NOALIGNMENTFAULTEXCEPT
from sys import stdout
from clingo import Control, Model, Symbol
from time import perf_counter
from typing import List
from os import path 
from solution import Solution, Plan
from argparse import ArgumentParser, Namespace
import logging

"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')
PREPROCESSING_FILE = path.join(ENCODING_DIR,'setup.lp')
POSTPROCESSING_FILE = path.join(ENCODING_DIR,'position_to_occurs.lp')
SOLVE_VERTEX_CL_FILE = path.join(ENCODING_DIR,'solve_vertex_cl.lp')
SOLVE_EDGE_CL_FILE = path.join(ENCODING_DIR,'solve_edge_cl.lp')
CONFLICT_DETECTION_FILE = path.join(ENCODING_DIR,'conflict_detection.lp')

"""Logging setup"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(message)s')
handler = logging.StreamHandler(stdout)
handler.setFormatter(formatter)
logger.addHandler(handler)

class IterativeSolver:
    """
    Implements a solver object which executes the iterative solving algorithm.

    Attributes:
        instance_file : str
            Path to the asprilo instance file that is to ve solved.
        edge_iter : int
            Decreasing value for how many edge iterations are allowed to be done
        max_edge_iter : int
            Maximum number of edge iterations allowed to be done
        vertex_iter : int
            Decreasing value for how many vertex iterations are allowed to be done
        max_vertex_iter : int
            Maximum number of vertex iterations allowed to be done
        conflict_step: bool
            Signifier if the saved conflicts are current
        edge_cl_found: bool
            Signifier if in the conflict step, an edge collison was detected
        vertex_cl_found: bool
            Signifier if in the conflict step, a vertex collison was detected
        conflicts: List[Symbol]
            List of all currently found conflicts
        solution : Solution
            Stores the solution obtained by iterative solving.
    """

    def __init__(self,instance : str, edge_iter : int = 120, vertex_iter : int = 80, log_level : int = logging.INFO):
        
        self.instance_file = instance
        self.edge_iter = edge_iter
        self.max_edge_iter = edge_iter
        self.vertex_iter = vertex_iter
        self.conflict_step : bool = False
        self.edge_cl_found : bool = False
        self.vertex_cl_found : bool = False
        self.conflicts : List[Symbol] = []
        self.solution = Solution()
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

            for atom in model.symbols(shown=True):
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

    def postprocessing(self) -> None:
        """
        Search for conflict one last time to ensure satisfiability and update the data into an asprillo suitable form

        Side effects:
            self.conflicts. List of conflicts gets updated
            solution.plans : Plans for all agents are saved
            solution.cost: Cost of the plan gets calculated
            solution.makespan: Makespan of the plan gets calculated
            solution.satisfied: States if the final plan satisfies the constraints
        """

        ctl = Control(arguments=["-Wnone"])

        ctl.load(CONFLICT_DETECTION_FILE)

        with ctl.backend() as backend:
            for plan in self.solution.plans.values():
                for atom in plan.positions:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.conflict_parser)


        def postprocessing_parser(model : Model) -> bool:

            for atom in model.symbols(shown=True):
                agent : int = atom.arguments[0].arguments[1].number
                self.solution.plans[agent].occurs.append(atom)

            return False

        logger.debug("Postprocessing invoked")

        ctl : Control = Control(["-Wnone"])

        ctl.load(POSTPROCESSING_FILE)

        with ctl.backend() as backend:
            for plan in self.solution.plans.values():
                for atom in plan.positions:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])

        ctl.ground([("base",[])])

        ctl.solve(on_model=postprocessing_parser)

        for plan in self.solution.plans.values():
            cost : int = len(plan.positions)-1
            self.solution.cost += cost
            self.solution.makespan = max(self.solution.makespan,cost)
            plan.cost = cost

        self.solution.satisfied =not( self.vertex_cl_found or self.edge_cl_found )

    def solve(self) -> Solution:
        """
        Executes the iterative solving algorithm.

        Side effect: 
            Updates plans in solution and execution time. 

        Returns:
            Solution object with the solution obtained by iterative solving.
        """


        logger.debug("Solve invoked")
        start_t : float = perf_counter()

        self.preprocessing()
        self.solution.plans = deepcopy(self.solution.get_initial_plans())

        self.solve_edge()
        self.solve_vertex()

        self.postprocessing()
        self.solution.execution_time = perf_counter()-start_t
        logger.debug("Solving finished")

        return self.solution

    def solve_edge(self) -> None:
        """
        Searches for conflicts and if edge conflicts are present, solves them, if edge iterations are left.
        Only stops once no edge conflict is found or no iterations are left.

        Side effect: 
            Updates plans in solution, updates list of conflicts, decreases edge_iter.
        """

        logger.debug("SolveEdge invoked")

        ctl : Control

        while self.edge_iter > 0:

            if not self.conflict_step:

                self.conflict_step = True

                ctl = Control(arguments=["-Wnone"])

                ctl.load(CONFLICT_DETECTION_FILE)

                with ctl.backend() as backend:
                    for plan in self.solution.plans.values():
                        for atom in plan.positions:
                            fact = backend.add_atom(atom)
                            backend.add_rule([fact])

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.conflict_parser)

            self.edge_iter = self.edge_iter -1

            #if an edge collision was found
            if self.edge_cl_found:
                self.conflict_step = False

                #solve edge collision
                ctl = Control(arguments=["-Wnone"])

                ctl.load(SOLVE_EDGE_CL_FILE)

                with ctl.backend() as backend:
                    for atom in self.solution.instance_atoms:
                        fact = backend.add_atom(atom)
                        backend.add_rule([fact])
                    for plan in self.solution.plans.values():
                        for atom in plan.positions:
                            fact = backend.add_atom(atom)
                            backend.add_rule([fact])
                    for atom in self.conflicts:
                        fact = backend.add_atom(atom)
                        backend.add_rule([fact])

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.model_solving_parser)


                #repeat
            
            #if no edge collision was found, end edge iteration
            else:
                logger.debug(f'SolveEdge terminated after {self.max_edge_iter - self.edge_iter} iterations')
                return

        logger.debug('Edge iterations exeeded')

    def solve_vertex(self):
        """
        Searches for conflicts and if vertex conflicts are present, solves them, if vertex iterations are left.
        If an edge conflict gets found, pause solve_vertex and handle solve_edge if iterations are left.
        Only stops if vertex iterations ran out, or no conflicts are left

        Side effect: 
            Updates plans in solution, updates list of conflicts, decreases vertex_iter.
        """

        logger.debug("SolveVertex invoked")
        
        while self.vertex_iter > 0:

            if not self.conflict_step:

                self.conflict_step = True
                
                ctl = Control(arguments=["-Wnone"])

                ctl.load(CONFLICT_DETECTION_FILE)

                with ctl.backend() as backend:
                    for plan in self.solution.plans.values():
                        for atom in plan.positions:
                            fact = backend.add_atom(atom)
                            backend.add_rule([fact])

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.conflict_parser)

            if(self.edge_cl_found and self.edge_iter > 0):
                logger.debug("SolveVertex created edge collision")                
                self.solve_edge()
                continue


            self.vertex_iter = self.vertex_iter -1

            #if an vertex collision was found
            if self.vertex_cl_found:
                self.conflict_step=False
                #solve vertex collision
                ctl = Control(arguments=["-Wnone"])

                ctl.load(SOLVE_VERTEX_CL_FILE)

                with ctl.backend() as backend:
                    for atom in self.solution.instance_atoms:
                        fact = backend.add_atom(atom)
                        backend.add_rule([fact])
                    for plan in self.solution.plans.values():
                        for atom in plan.positions:
                            fact = backend.add_atom(atom)
                            backend.add_rule([fact])
                    for atom in self.conflicts:
                        fact = backend.add_atom(atom)
                        backend.add_rule([fact])
                ctl.ground([("base",[])])

                ctl.solve(on_model=self.model_solving_parser)

                #repeat
            else:
                logger.debug("SolvingVertex terminated")
                break

        logger.debug('Vertex iterations exeeded')
               
    def conflict_parser(self, model : Model) -> bool:
        self.edge_cl_found = False
        self.vertex_cl_found = False
        self.conflicts = []
        for atom in model.symbols(shown=True):
            
            if str(atom.name) == 'position':
                agent : int = atom.arguments[0].number
                self.solution.plans[agent].positions.append(atom)
            elif atom.arguments[0].name == 'edge':

                self.edge_cl_found = True
                self.conflicts.append(atom)
            elif atom.arguments[0].name == 'vertex':

                self.vertex_cl_found = True
                self.conflicts.append(atom)

        return False

    def model_solving_parser(self,model : Model) -> bool:
        self.solution.clear_plans()
        for atom in model.symbols(shown=True):
            self.solution.plans[atom.arguments[0].number].positions.append(atom)
        return False
            

if __name__ == "__main__":
    """Command line argument parsing"""
    parser = ArgumentParser()
    parser.add_argument("instance", type=str,help="Path to asprilo instance file")
    parser.add_argument("--edgeIterations",default=30, type=int,help="Set the maximum amount of edge solvings to be done, default 30")
    parser.add_argument("--vertexIterations",default=30, type=int,help="Set the maximum amount of vertex solvings to be done, default 30")
    parser.add_argument("-b", "--benchmark", default=False, action="store_true",help="Outputs execution time and solution statistics.")
    parser.add_argument("-d", "--debug", default=False, action="store_true",help ="Makes solving process verbose for debugging purposes.")


    args : Namespace = parser.parse_args()
    solution : Solution = IterativeSolver(args.instance,args.edgeIterations,args.vertexIterations,logging.DEBUG if args.debug else logging.INFO).solve()
    
    solution.save("plan.lp")
    
    if args.benchmark:
        logger.info(f'Execution time : {solution.execution_time:.2f}s')
        logger.info(f'Sum of costs : {solution.get_soc()}')
        logger.info(f'Makespan : {solution.get_makespan()}')
        logger.info(f'Total moves : {solution.get_total_moves()}')