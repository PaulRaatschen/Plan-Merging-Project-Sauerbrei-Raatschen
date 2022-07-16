import string
from typing import List, Symbol, Dict, Union
from math import inf

class Solution:

    def __init__(self,agents : List[int] = None,inits : List[Symbol] = None, instance_atoms : List[Symbol] = None, plans : Dict[int,Dict[str,Union[List[Symbol],int]]] = None, max_horizon : int = 0):
        self.agents = agents if agents else []
        self.inits = inits if inits else []
        self.instance_atoms = instance_atoms if instance_atoms else []
        self.plans = plans if plans else {}
        self.max_horizon = max_horizon
        self.cost : int = 0
        self.execution_time : float = 0.0



    def clear_plans(self) -> None:
        self.plans = {}

    def clear_plan(self,agent : int) -> None:
        if agent in self.plans:
            self.plans[agent] = {'occurs' : [],'positions' : [], 'cost' : inf}

    def save(self, filepath : string):

        with open(filepath, 'w', encoding='utf-8') as file:
            for init in self.inits:
                file.write()
