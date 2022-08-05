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
        self.singleAgentPF = path.join(ENCODING_DIR,'singleAgentPF.lp')
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


    def solve(self):

        ctl = Control(arguments=["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(self.setup)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.standard_parser)

        for robot in self.robots:
            if(self.verbose):
                print("Generate Path for Robot: "+ robot)
            ctl = Control(["--opt-mode=opt",f"-c id={robot}","-c horizon="+str(self.result.max_horizon),"-Wnone"])
            ctl.load(self.instance_file)
            ctl.load(self.singleAgentPF)
            
            ctl.ground([("base",[])])
            
            ctl.solve(on_model=self.singleAgentPF_parser)

        ctl = Control(arguments=["-Wnone"])

        for atom in self.individualRobotPaths:
            ctl.add("base",[],f"{atom}.")

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