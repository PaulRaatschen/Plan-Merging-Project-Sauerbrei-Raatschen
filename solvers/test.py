
from os import path
from iterative_solving import IterativeSolver
from prioritized_planning import PrioritizedPlanningSolver
from solution import Solution
from clingo import Symbol, Function, Control, Model, Number
import logging

from cbs import CBSSolver


"""Directorys and asp files"""
WORKING_DIR : str = path.abspath(path.dirname(__file__))
INSTANCE_FILE : str = path.join(WORKING_DIR,r'..\naiveplanning\instances\12r_CrossRoad.lp')
MAPF_FILE : str = path.join(WORKING_DIR,r'.\encodings\multiAgentPF_inc.lp')
PREPROCESSING_FILE : str = path.join(WORKING_DIR,r'.\encodings\setup.lp')


sol : Solution = CBSSolver(INSTANCE_FILE,log_level=logging.DEBUG).solve()

for i in [1,2,9,10]:
    for atom in sol.plans[i]['positions']:
        print(f'{atom}.')


print(sol.satisfied)
print(f'Execution time : {sol.execution_time:.3}s\n sum of costs : {sol.get_soc()}\n normalized : {sol.get_norm_soc():.3}\n makespan: {sol.get_makespan()}\n normalized : {sol.get_norm_makespan():.3}\n total moves : {sol.get_total_moves()}\n normalized : {sol.get_norm_total_moves():.3}\n density : {sol.get_density():.3}')


sol.save(r'sol.lp')
'''
def preprocessing(solution : Solution) -> None:

        def preprocessing_parser(model : Model, solution : Solution) -> bool:

            for atom in model.symbols(atoms=True):
                if(atom.name == 'init'):
                    solution.inits.append(atom)
                elif(atom.name == 'numOfRobots'):
                    solution.agents = list(range(1,atom.arguments[0].number+1))
                elif(atom.name == 'numOfNodes'):
                    solution.num_of_nodes = atom.arguments[0].number
                else:
                    solution.instance_atoms.append(atom)

            return False

        ctl : Control = Control(["-Wnone"])

        ctl.load(INSTANCE_FILE)

        ctl.load(PREPROCESSING_FILE)

        ctl.ground([("base",[])])

        ctl.solve(on_model=lambda model : preprocessing_parser(model,solution))


sol = Solution()

ctl = Control()

meta_agent = (1,2)

ctl.load(MAPF_FILE)

with ctl.backend() as backend:
    for atom in sol.instance_atoms:
        fact = backend.add_atom(atom)
        backend.add_rule([fact])

        for agt in meta_agent:
            act = backend.add_atom(Function(name='planning',arguments=[Number(agt)]))
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
                    for model in handle:
                        pass
                    optimal_model = model
                    for atom in optimal_model.symbols(shown=True):
                        if atom.name == 'occurs':
                            self.plans[atom.arguments[0].number]['occurs'].append(atom)
                        elif atom.name == 'goalReached':
                            agt : int = atom.arguments[0].number 
                            self.plans[agt]['cost'] = atom.arguments[1].number - old_costs[agt]
                            self.plans[agt]['positions'].append(atom)
                        else:
                            self.plans[atom.arguments[0].number]['positions'].append(atom)
                        
        return ret.satisfiable



sol.save(r'sol.lp')

'''




