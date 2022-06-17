# Plan-Merging-Project-Sauerbrei-Raatschen

Student project at the University of Potsdam about multi agent path finding through plan merging in ASP (Answer Set Programming) using the [asprilo](https://potassco.org/asprilo/) framework.

# Comparison of naive planning and prioritized planning 



## Naive Planning

### Strengths

+ simple implementation and idea
+ solves most instances

### Weaknesses

- runs an undefiend number of iterations
- generated paths contain unnecessary, sometimes other robot blocking, moves
- won't generate optimal paths for robots

### Prioritized Planning

### Strengths

+ still simple implementation and idea
+ solves most instances, more then the naive variant
+ even when the program can't solve an instance, it solves as many conflicts between agents as possible

### Weaknesses

- priority given to the robots can cause the instance to become unsolvable


## Contributers

- Nico Sauerbrei: nico.sauerbrei@uni-potsdam.de
- Paul Raatschen: raatschen@uni-potsdam.de


