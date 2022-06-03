# Naive Planning

## Running The Encodings

To run the encodings one simply has to run the pyhon file `ConflictSolver.py`, with the instance he wants to solve as an input.
It is recommended to run the programm in the directory directly for example:

python encodings\ConflictSolver.py instances\Cross2rB.lp

This will create a directory named Cross2rB in your current directoy, containing the generated unsolved paths of the agents ("Paths-Cross2rB.lp"), as well as a file containing the new solved path ("NewPlan-Cross2rb.lp").

Furthermore, if installed correctly the Asprillo visualizer will open itself, with the newly generated plan opened.

## Directories

- `instances` contains M-domain asprilo instances to test our conflict resolution approaches 
- `encodings` contains all ASP encodings aiming to resolve the plan merging problem
- `solved` contains the solved versions of the instances in `instances`
- `failed` contains the plans of instances, which weren't solvable by the encodings
