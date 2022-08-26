# Plan-Merging-Project

## About

This is a student project at the University of Potsdam about multi agent path finding through plan merging in ASP (Answer Set Programming) using the [asprilo](https://potassco.org/asprilo/) framework. We implement three different plan merging approaches

## Directories
- `Benchmarks/` contains the asprilo instance files used for benchmarking.
- `Solvers/` contains our plan merging implementations, asp encodings and additional python scripts for benchmarking.
- `Report/` contains our project report.
- `Presentation/` contains the slides for our project presentation.

## Iterative Solving


## Prioritized Planning


## Conflict Based Search

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

## Contributers

- Nico Sauerbrei: nico.sauerbrei@uni-potsdam.de
- Paul Raatschen: raatschen@uni-potsdam.de


