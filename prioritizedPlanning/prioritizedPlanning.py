from pickletools import optimize
from clingo import Control, Number, Function
from time import time
from numpy import round
import argparse

class PrioritizedPlanner:

    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("instance", type=str)

        parser.add_argument("-b", "--benchmark", default=False, action="store_true")

        parser.add_argument("-o", "--optimize", default=False, action="store_true")

        parser.add_argument("-v", "--verbose", default=False, action="store_true")

        args = parser.parse_args()
        
        self.iteration = 0

        self.model = []

        self.plans = {}

        self.robots = []

        self.preprocessing_facts = []

        self.position_facts = []

        self.preprocessing_file = "./setup.lp"

        self.sapf_file = "./incrementalPathfinding.lp"

        self.optimize_file = "./optimize.lp"

        self.nocl_sapf_file = "./incrementalPathfinding_nocl.lp"

        self.conflict_detection_file = "./conflict_detection.lp"

        self.instance_file = args.instance

        self.benchmark = args.benchmark

        self.optimize = args.optimize

        self.verbose = args.verbose

        if self.benchmark:
            time_start = time()
            self.solve()
            time_end = time()

            print(f"Execution Time: {round(time_end-time_start,3)} seconds")
            total_moves = 0
            for robot in self.robots:
                if self.plans[robot][1]:
                    moves = self.plans[robot][1][0]
                    print(f"Robot: {robot}, Moves: {moves}")
                    total_moves += moves
                else:
                    print(f"No solution for Robot {robot}")
            print(f"Total moves: {total_moves}")
        else:
            self.solve()

    def preprocessing_parser(self,model):

        if not model:
            return

        for atom in model.symbols(atoms=True):
            if(atom.name == "init"):
                self.model.append(atom)

            elif(atom.name == "numOfRobots"):
                    self.robots = list(range(1,atom.arguments[0].number+1))

            else:
                self.preprocessing_facts.append(atom)

    def nocl_parser(self,model):

        if not model:
            return

        for atom in model.symbols(shown=True):
            if(atom.name == "rPosition"):
                self.position_facts.append(atom)
            
    def optimization_parser(self,model,pathlengths):

        if not model:
            return

        ordering = [(0,0,0)] * len(self.robots)

        for atom in model.symbols(shown=True):

            robot = atom.arguments[0].number

            ordering[robot-1] = (robot,atom.arguments[1].number,pathlengths[robot-1])

        if ordering[0][0] == 0:
            self.robots = list(map(lambda x : x[0], sorted([(robot,pathlengths[robot-1]) for robot in self.robots], key=lambda x : x[1])))
        else:
            self.robots = list(map(lambda x : x[0], sorted(ordering, key=lambda x : (x[1],x[2]))))

    def main_parser(self,model,robot):
        if not model:
            return

        for atom in model.symbols(shown=True):
            if(atom.name == "rPosition"):
                self.position_facts.append(atom)
            else:
                self.plans[robot][0].append(atom)
                self.model.append(atom)

    def solve(self):

        ctl = Control(arguments=["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(self.preprocessing_file)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.preprocessing_parser)

        if self.optimize:
            self.optimize_schedule()

        for robot in self.robots:

            if self.verbose:
                print(f"Solving for robot {robot}")

            self.plans[robot] = ([],[])

            self.plan_path(robot)

        with open("plan.lp",'w') as output:

            for atom in self.model:

                output.write(f"{atom}. ")

    def optimize_schedule(self):

        pathlengths = [0] * len(self.robots)

        imax = 60

        for robot in self.robots:

            ctl = Control(arguments=["-Wnone",f"-c r={robot}"])

            ctl.load(self.nocl_sapf_file)

            for atom in self.preprocessing_facts:
                ctl.add("base",[],f"{atom}.")

            pathlengths[robot-1] = self.iterative_solving(ctl,0,imax,lambda m : self.nocl_parser(m)) 

        ctl = Control(arguments=["-Wnone"])

        ctl.load(self.conflict_detection_file)

        for atom in self.preprocessing_facts:
                ctl.add("base",[],f"{atom}.")

        for atom in self.position_facts:
                ctl.add("base",[],f"{atom}.")

        ctl.ground([("base",[])])

        ctl.solve(on_model=lambda m : self.optimization_parser(m,pathlengths))

        self.position_facts = [] 


    def plan_path(self,robot):

        self.iteration = robot

        imax = 60
        
        ctl = Control(["--opt-mode=opt",f"-c r={robot}","-Wnone"])

        ctl.load(self.sapf_file)

        for atom in self.preprocessing_facts:
            ctl.add("base",[],f"{atom}.")

        for atom in self.position_facts:
            ctl.add("base",[],f"{atom}.")

        result = self.iterative_solving(ctl,0,imax,lambda m : self.main_parser(m,self.iteration))

        if result <= imax:
            self.plans[robot][1].append(result)


    def iterative_solving(self,ctl,imin,imax,model_parser):

        ctl.add("check", ["t"], "#external query(t).")

        ret, step = None, 0

        while((step < imax) and (ret is None or step < imin or (not ret.satisfiable))):

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

        return step + 1 if (ret and (not ret.satisfiable)) else step - 1


if __name__ == "__main__":
    PrioritizedPlanner()