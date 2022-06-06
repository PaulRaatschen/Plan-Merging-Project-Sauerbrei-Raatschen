from clingo import Control, Number, Function
import argparse

class PrioritizedPlanner:

    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("instance", type=str)

        args = parser.parse_args()
        
        self.iteration = 0

        self.model = []

        self.plans = {}

        self.robots = []

        self.preprocessing_facts = []

        self.position_facts = []

        self.preprocessing_file = "./setup.lp"

        self.sap_file = "./incrementalPathfinding.lp"

        self.instance_file = args.instance

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

    def main_parser(self,model,robot):
        if not model:
            return

        for atom in model.symbols(shown=True):
            if(atom.name == "rPosition"):
                self.position_facts.append(atom)
            else:
                self.plans[robot].append(atom)
                self.model.append(atom)

    def solve(self):

        ctl = Control(arguments=["-Wnone"])

        ctl.load(self.instance_file)

        ctl.load(self.preprocessing_file)

        ctl.ground([("base",[])])

        ctl.solve(on_model=self.preprocessing_parser)

        for robot in self.robots:

            self.plans[robot] = []

            self.plan_path(robot)

        with open('plan3.lp','w') as output:

            for atom in self.model:

                output.write(f"{atom}. ")

    def plan_path(self,robot):
        
        ctl = Control(["--opt-mode=opt",f"-c r={robot}","-Wnone"])

        ctl.load(self.sap_file)

        for atom in self.preprocessing_facts:
            ctl.add("base",[],f"{atom}.")

        for atom in self.position_facts:
            ctl.add("base",[],f"{atom}.")

        ctl.add("check", ["t"], "#external query(t).")

        imax = 50
        imin = 0
  
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
            ret, step = ctl.solve(on_model=lambda m : self.main_parser(m,robot)), step + 1   

if __name__ == "__main__":
    PrioritizedPlanner()