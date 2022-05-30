import clingo
import argparse

class prioritizedPlanner:

    def __init__(self):

        parser = argparse.ArgumentParser()

        parser.add_argument("instance", type=str)

        args = parser.parse_args()
        
        self.iteration = 0

        self.model = []

        self.plans = {}

        self.robots = []

        self.positions = ""

        self.preprocessing = "./setup.lp"

        self.sap = "./singleAgentPlanning.lp"

        self.instance = args.instance

        self.main()


    def main(self):

        ctl_pre = clingo.Control()

        preprocessing_facts = ""

        ctl_pre.load(self.instance)

        ctl_pre.load(self.preprocessing)

        ctl_pre.ground([("base",[])])

        with ctl_pre.solve(yield_=True) as handle:

            model = handle.model()

            for atom in model.symbols(shown=True):

                if(atom.name == "init"): self.model.append(atom)

                else:
                    preprocessing_facts += f"{atom}. "

                if(atom.name == "numOfRobots"):
                    self.robots = list(range(1,atom.arguments[0].number+1))

        for robot in self.robots:

            print(robot)

            self.plans[robot] = []

            clt_main = clingo.Control(["0","--opt-mode=opt"])

            clt_main.load(self.sap)

            clt_main.add("base",[],preprocessing_facts)

            if self.positions : clt_main.add("base",[],self.positions)

            clt_main.ground([("base",[]),("robot",[clingo.Number(robot)])])

            with clt_main.solve(yield_=True) as handle:

                model = handle.model()

                for atom in model.symbols(shown=True):

                    if(atom.name == "occurs"): 

                        self.plans[robot].append(atom)

                        self.model.append(atom)

                    if(atom.name == "rPosition"): self.positions += f"{atom}. "

            self.iteration += 1

        with open('plan.lp','w') as output:

            for atom in self.model:

                output.write(f"{atom}.")

if __name__ == "__main__":
    prioritizedPlanner()