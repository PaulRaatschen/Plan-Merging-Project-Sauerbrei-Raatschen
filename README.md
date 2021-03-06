# Plan-Merging-Project-Sauerbrei-Raatschen

## About

Student project at the University of Potsdam about multi agent path finding through plan merging in ASP (Answer Set Programming) using the [asprilo](https://potassco.org/asprilo/) framework.

## Directories

The directories contain different solving approaches for the multiple agent path finding problem.

### Naive Planning

As the name suggest naive planning contains a naive approach of solving the conflict, here the conflict between two robots at a time is solved, causing "phantom" moves from earlier solving to get carried over, which became unneccesary as a the original conflict already was solved by a different conflict from one of the robots with another.

### Prioritized Planning

In prioritized planning paths get assigned iterativly, by creating one after the other, non conflicting paths.

For example:
- robot 1 gets his path assigned
- robot 2 gets his path assigned, avoiding collision with robot 1
- robot 3 gets his path assigned, yet avoiding collision with robot 1 and robot 2

## Contributers

- Nico Sauerbrei: nico.sauerbrei@uni-potsdam.de
- Paul Raatschen: raatschen@uni-potsdam.de


