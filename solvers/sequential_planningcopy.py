from clingo import Control, Number, Function
from time import time
from os import path
import argparse
import solution

class SequentialPlanner:

    def __init__(self,args):
        
        self.iteration = 0

        self.model = []

        self.plans = {}

        self.robots = []

        self.standard_facts = []

        self.individualRobotPaths = []

        """Directorys and asp files"""
        WORKING_DIR : str = path.abspath(path.dirname(__file__))
        ENCODING_DIR : str = path.join(WORKING_DIR,'encodings')


        self.setup = path.join(ENCODING_DIR,'setup.lp')
        self.singleAgentPF = path.join(ENCODING_DIR,'singleAgentPF_inc.lp')
        self.collisionToRPosition = path.join(ENCODING_DIR,'collisionToRPosition.lp') 
        self.postprocessing_file = path.join(ENCODING_DIR,'rPositionToCollision.lp')

        self.edgeSolver_file = path.join(ENCODING_DIR,'collision_evasion.lp')
        self.vertexSolver_file = path.join(ENCODING_DIR,'collision-avoidance-wait.lp')

        self.conflict_detection_file = path.join(ENCODING_DIR,'conflict_detection.lp')

        self.instance_file = args.instance

        self.edgeIterations = args.edgeIterations
        self.maxEdgeIterations = args.edgeIterations
        self.vertexIterations = args.vertexIterations
        self.ConflictStep = False
        self.edgeCollisionFound = False
        self.result = solution.Solution()

        self.benchmark = args.benchmark

        self.verbose = args.verbose
        time_start = time()
        self.solve()
        time_end = time()

        self.result.execution_time = time_end-time_start
        if(self.edgeIterations == 0 or self.vertexIterations == 0):
            self.result.satisfied = False
        else:
            self.result.satisfied = True



        if self.benchmark:


            print(f"Execution Time: {time_end-time_start:.3f} seconds")

    def incremental_solving(self, ctl : Control, max_horizon : int, model_parser) -> None:

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



    def plan_path(self, agent : int) -> bool:

        ctl : Control
        cost : int
        old_cost : int = 0

        self.result.plans[agent] = {'occurs' : [],'positions' : [], 'cost' : 0}


        if(self.verbose):
            print("Planning for agent ",agent )

        def plan_path_parser(model, agent : int, solution) -> bool:
            for atom in model.symbols(shown=True):
                    if(atom.name == 'occurs'):
                        solution.plans[agent]['occurs'].append(atom)
                    else:
                        solution.plans[agent]['positions'].append(atom)
            return False

        ctl = Control(arguments=['-Wnone',f'-c r={agent}'])

        ctl.load(self.singleAgentPF)

        with ctl.backend() as backend:
            for atom in self.standard_facts:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])


        cost = self.incremental_solving(ctl,self.result.max_horizon,lambda model : plan_path_parser(model,agent,self.result))


        if(self.verbose):
            print("Finished solving for agent ", agent)

        
    def solve(self):

        ctl = Control(arguments=["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(self.setup)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.standard_parser)

        for robot in self.robots:
            self.plan_path(robot)

        ctl = Control(arguments=["-Wnone"])

        with ctl.backend() as backend:
            for plan in self.result.plans.values():
                for atom in plan['occurs']:
                    fact = backend.add_atom(atom)
                    backend.add_rule([fact])


        ctl.load(self.instance_file)
        ctl.load(self.collisionToRPosition)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.standard_parser)



        self.solveEdge()

        self.solveVertex()
        if(self.verbose):
            for atom in self.standard_facts:
                print(atom)
            print("\n")

        ctl = Control(arguments=["-Wnone"])

        for atom in self.standard_facts:
            ctl.add("base",[],f"{atom}.")

        ctl.load(self.postprocessing_file)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.standard_parser)

        self.result.max_horizon = 0
        for atom in self.standard_facts:
                if atom.name == "occurs":
                    self.result.cost += 1
                    if( atom.arguments[2].number > self.result.max_horizon):
                        self.result.max_horizon = atom.arguments[2].number

        with open("plan.lp",'w') as output:
            for atom in self.model:


                output.write(f"{str(atom)}. ")

            for atom in self.standard_facts:
                output.write(f"{str(atom)}. ")
        return self.result
    
    def standard_parser(self,model):

        if not model:
            return
        
        self.standard_facts.clear()
        for atom in model.symbols(shown=True):
            if(atom.name == "init"):
                self.model.append(atom)
                self.result.instance_atoms.append(atom)
            elif(atom.name == 'numOfNodes'):
                self.result.max_horizon = atom.arguments[0].number * 2

            elif(atom.name == "numOfRobots"):
                                
                self.robots = list(range(1,atom.arguments[0].number+1))

            else:
                self.standard_facts.append(atom)
    
    
    def singleAgentPF_parser(self,model):

        if not model:
            return
        
        for atom in model.symbols(shown=True):
            if(atom.name == "occurs"):
                self.individualRobotPaths.append(atom)


    
    
    def solveEdge(self):
        
        while(self.edgeIterations > 0):
            if(self.ConflictStep == False):
                self.ConflictStep = True
                ctl = Control(arguments=["-Wnone"])

                ctl.load(self.conflict_detection_file)
                ctl.load(self.instance_file)

                for atom in self.standard_facts:
                    
                    ctl.add("base",[],f"{atom}.")

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.standard_parser)

                self.edgeCollisionFound = False
                for atom in self.standard_facts:
                    if(atom.name == "edgeCollision"):
                        self.edgeCollisionFound = True

            self.edgeIterations = self.edgeIterations -1

            #if an edge collision was found
            if self.edgeCollisionFound:
                self.ConflictStep = False

                #solve edge collision
                ctl = Control(arguments=["-Wnone"])

                ctl.load(self.edgeSolver_file)

                for atom in self.standard_facts:
                    ctl.add("base",[],f"{atom}.")

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.standard_parser)


                #repeat
            
            #if no edge collision was found, end edge iteration
            else:
                if(self.verbose):
                    print("Edge, stopped after iteration " + str(self.maxEdgeIterations - self.edgeIterations))

                return
    def solveVertex(self):
        
        while(self.vertexIterations > 0):

            if(self.ConflictStep == False):
                self.ConflictStep = True
                
                ctl = Control(arguments=["-Wnone"])

                ctl.load(self.conflict_detection_file)
                ctl.load(self.instance_file)

                for atom in self.standard_facts:
                
                    ctl.add("base",[],f"{atom}.")

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.standard_parser)

            vertexCollisionFound = False
            for atom in self.standard_facts:
                if(atom.name == "vertextCollision"):
                    vertexCollisionFound = True
                if(atom.name == "edgeCollision" and self.edgeIterations > 0):
                    if(self.verbose):
                        print("Edgecollision found in Vertexcollision\n")
                    self.solveEdge()

            self.vertexIterations = self.vertexIterations -1

            #if an vertex collision was found
            if vertexCollisionFound:
                self.ConflictStep=False
                #solve vertex collision
                ctl = Control(arguments=["-Wnone"])

                ctl.load(self.vertexSolver_file)

                for atom in self.standard_facts:
                    ctl.add("base",[],f"{atom}.")

                ctl.ground([("base",[])])

                ctl.solve(on_model=self.standard_parser)

                #repeat
            else:
                break
def benchmark(instancePath):
    args = argparse.Namespace()
    args.instance = instancePath
    args.edgeIterations = 80
    args.vertexIterations = 120
    args.benchmark = False
    args.verbose = False

    return SequentialPlanner(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("instance", type=str)
    parser.add_argument("edgeIterations",default=10, type=int)
    parser.add_argument("vertexIterations",default=20, type=int)

    parser.add_argument("-b", "--benchmark", default=False, action="store_true")

    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    args = parser.parse_args()
    SequentialPlanner(args)