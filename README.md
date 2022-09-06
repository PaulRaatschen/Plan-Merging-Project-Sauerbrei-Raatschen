# Plan-Merging-Project

## About

This is a student project at the University of Potsdam about multi agent path finding through plan merging in ASP (Answer Set Programming) using the [asprilo](https://potassco.org/asprilo/) framework. We implement three different plan merging approaches

## Directories
- `Benchmarks/` contains the asprilo instance files used for benchmarking, as well as an copy of our used benchmarking data, and an associated Jupyter Notebook .
- `Solvers/` contains our plan merging implementations, asp encodings and additional python scripts for benchmarking.
- `Report/` contains our project report.
- `Presentation/` contains the slides for our project presentation.
- `INSTANCES_FOR_OTHER_GROUPS/` contains the asprillo instance files for the benchmarking from other groups.

## Iterative Solving

Plan merging approach based on iterativly solving edge and vertex conflicts.

File: iterative_solving.py

## Prioritized Planning

Plan merging by avoiding conflicts through a priority order of agents. 

File: prioritized_planning.py

## Conflict Based Search

ASP based implementation of the [conflict based search](https://www.sciencedirect.com/science/article/pii/S0004370214001386) MAPF algorithm by Sharon et al.

File: cbs.py

## Usage
To run a solver on an asprilo instance: 
(Tested with Python 3.9, requires [clingo](https://potassco.org/clingo/) package)
```bash
python <SOLVER>.py <INSTANCE> 
```
For a list of all available options:
```bash
python <SOLVER>.py --help
```

## Benchmark Programs

### Instance Generator

Generates instances based on flags.

File: generate_instance.py
```bash
python generate_instance.py <WIDTH> <HEIGHT> <NUMBEROGROBOTS> <MAPTYPE>
```
mapType:
        "Rooms": Generates Rooms, "-nRooms=<x>" must be specified
        "Grid": Generates an instance in a grid like pattern, "-hLines=<x>" must be specified for number of horizontal lines, "-vLines=<x>" for vertical lines.
        "Random": Blocks random nodes, yet with every goal still accessable by each agent, "-nWalls <x>" must be specified.

### Benchmarker

Runs each solver over an instance, or over multiple generated instances.
How new instance should be generated must be defined in the pythonfile itself.

(Requieres [pandas](https://pandas.pydata.org/docs/index.html))
File: benchmarker.py
```bash
python benchmarker.py <INSTANCE>
```
```bash
python benchmarker.py <NAMETOCATEGORIZEINSTANCE> -g
```

## Contributers

- Nico Sauerbrei: nico.sauerbrei@uni-potsdam.de
- Paul Raatschen: raatschen@uni-potsdam.de


