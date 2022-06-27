from queue import PriorityQueue
from clingo import Control, Number, Function
from time import perf_counter
from os import path
from argparse import ArgumentParser
from sys import exit
from math import inf



WORKING_DIR = path.abspath(path.dirname(__file__))
ENCODING_DIR = path.join(WORKING_DIR,'Encodings')
PREPROCESSING_FILE = path.join(WORKING_DIR,'setup.lp')
SAPF_FILE = path.join(ENCODING_DIR,'singleAgentPF_iterative.lp')
VALIADTION_FILE = path.join(ENCODING_DIR,'validate.lp')

plan = []
agents = []
preprocessing_atoms = []
max_horizon = 0

parser = ArgumentParser()

parser.add_argument("instance", type=str)

parser.add_argument("-b", "--benchmark", default=False, action="store_true")

parser.add_argument("-o", "--optimize", default=False, action="store_true")

args = parser.parse_args()

INSTANCE_FILE = args.instance



class CTNode:

    def __init__(self,solution = None, constraints= None):
        self.solution = solution if solution else {}
        self.constraints = constraints if constraints else []
        self.cost = 0
            
    def low_level(self,agent):

        old_cost = self.plans[agent][2] if self.plans[agent] else 0

        def low_level_search_parser(model):
            self.plans[agent] = [[],[],0]
            for atom in model.symbols(shown=True):
                if(atom.name == 'occurs'):
                    self.plans[agent][0].append(atom)
                else:
                    self.plans[agent][1].append(atom)

        ctl = Control(['-Wnone',f'-c r={agent}'])

        ctl.load(SAPF_FILE)

        with ctl.backend() as backend:
            for atom in preprocessing_atoms + self.constraints:
                fact = backend.add_atom(atom)
                backend.add_rule([fact])

        imin, imax = 0, max_horizon

        step, ret = 0, None

        while ((step < imax) and (ret is None or (not ret.satisfiable))):
                    parts = []
                    parts.append(("check", [Number(step)]))
                    if step > 0:
                        ctl.release_external(Function("query", [Number(step - 1)]))
                        parts.append(("step", [Number(step)]))
                    else:
                        parts.append(("base", []))

                    ctl.ground(parts)
                    ctl.assign_external(Function("query", [Number(step)]), True)
                    ret, step = ctl.solve(on_model=low_level_search_parser), step + 1

        agent_cost = inf if (ret and (not ret.satisfiable)) else step - 1

        self.solution[agent][2] = agent_cost

        self.cost += (agent_cost-old_cost)

    def validate_plans(self):

        def verify_parser(self,model,conflicts):
            for atom in model.symbols(shown=True):
                conflicts.append(atom)

        ctl = Control(['-Wnone'])

        ctl.load(VALIADTION_FILE)

        with ctl.backend() as backend:
            for plan in self.solution.values():
                for instance in plan:
                    fact = backend.add_atom(instance[1])
                    backend.add_rule([fact])

        ctl.ground([('base',[])])

        conflicts = []

        ctl.solve(on_model=lambda model : verify_parser(self,model,conflicts))

        return conflicts

    def branch(self,conflict):

        node1 = CTNode(self.solution,self.constraints)
        node2 = CTNode(self.solution,self.constraints)

        ctype = conflict.arguments[0].string
        agent1 = conflict.arguments[1]
        agent2 = conflict.arguments[2]
        time = conflict.arguments[4]

        if(ctype == 'vertex'):
            loc = conflict.arguments[3]
            node1.constraints.append(Function(name="constraint", arguments=[agent1,loc,time]))
            node2.constraints.append(Function(name="constraint", arguments=[agent2,loc,time]))


        else:
            loc = conflict.arguments[3][0]
            move = conflict.arguments[3][1]
            node1.constraints.append(Function(name="constraint", arguments=[agent1,loc,move,time]))
            node2.constraints.append(Function(name="constraint", arguments=[agent2,loc,move,time]))

        node1.cost += node1.low_level(agent1.number)
        node2.cost += node2.low_level(agent2.number)

        return node1, node2

        
        
def preprocessing():

    def preprocessing_parser(model):
        for atom in model.symbols(atoms=True):
            if(atom.name == 'init'):
                plan.append(atom)
            elif(atom.name == 'numOfRobots'):
                agents = list(range(1,atom.arguments[0].number+1))
            elif(atom.name == 'numOfNodes'):
                max_horizon = atom.arguments[0].number
            else:
                preprocessing_atoms.append(atom)

        ctl = Control(["-Wnone"])

        ctl.load(INSTANCE_FILE)

        ctl.load(PREPROCESSING_FILE)

        ctl.ground([("base",[])])

        ctl.solve(on_model=preprocessing_parser(model))

def main():

        start_time = perf_counter()
    
        preprocessing()

        openQueue = PriorityQueue()
        
        solutionNodes = []

        root = CTNode()

        for agent in agents:
            low_level(agent)

        if(root.cost == inf):
            print("No initial solution found!")
            exit()
            
        openQueue.put((root.cost,root))

        while openQueue:

            p = openQueue.get()[1]

            conflicts = p.validate_plans()

            if conflicts:
                node1, node2 = p.branch(conflicts[0])
                openQueue.put((node1.cost,node1))
                openQueue.put((node2.cost,node2))
                
            else:
                solutionNodes.append(p)

        end_time = perf_counter()

        if solutionNodes:
            bestSolution = solutionNodes[0]

            with open("plan.lp",'w') as output:

                for atom in plan:
                    output.write(f"{atom}. ")
                for plan in bestSolution.solution.values():
                    for instance in plan:
                        output.write(f"{instance[0]}. ")

            print("Solution found")

            if args.benchmark:
                print(f'Execution Time: {end_time-start_time}s')

        else:
            print("No solution found")
    

if __name__ == '__main__':
    main()
