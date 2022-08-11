% Takes output of find_conflict.lp as input

% Determines ealiest conflict
earliestEdgeConflict(TMIN) :- TMIN = #min{T : conflict(edge,_,_,_,T)}.

{firstedgeConflict((X0,Y0),(X0',Y0'),T0,R,R'): conflict(edge,R,R',((X0,Y0),(X0',Y0')),T0), earliestEdgeConflict(T0)} = 1.

% Find an evasion step for one of the robots that frees the path for the other one

evasionStep((X,Y),T+1,R,D) :- firstedgeConflict((X0,Y0),(X0',Y0'),T0,R,R'),R != R', not position(R',(X,Y),_), backtraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|+1, node((X,Y)).
evasionStep((X,Y),T+1,R,D) :- firstedgeConflict((X0,Y0),(X0',Y0'),T0,R,R'),R != R', not position(R',(X,Y),_), backtraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|, node((X,Y)). 


% Initialize the backtraking from the conflict point 
backtraking(R,(X,Y),T0-1) :- firstedgeConflict((X0,Y0),(X0',Y0'),T0,R,R'), position(R,(X,Y),T0-1).

% Backtrack along the path (ideally only until an evasion step is found)
backtraking(R,(X,Y),T) :- backtraking(R,(X',Y'),T+1), position(R,(X,Y),T). % not evasion_step(_,T0,_,_) with T0 < T (dont know how to do in clingo)

% Determine the evasion step with the smallest distance to the conflict point in order to minimize waiting time
minDistance(DMAX) :- DMAX = #min{D : evasionStep((X,Y),T,R,D)}. 

% Choose one evasion step with minimal distance to the conflict point
{chosenStep((X,Y),T,R,D): evasionStep((X,Y),T,R,D)} = 1 :- minDistance(D). 

% Time that evading robot has to wait, either when he passes the point where the robot evaded, or when he passes the point where he had to stop
lwaitTime(W) :- chosenStep((X,Y),T0,R,D), position(R,(X',Y'),T0-1), R != R' ,position(R',(X',Y'),TD), W = TD - T0,firstedgeConflict(_,_,_,R,R').
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0+1), W = TD - T0,firstedgeConflict(_,_,_,R,R').
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0), W = TD - T0,firstedgeConflict(_,_,_,R,R').
lwaitTime(0).

% Longest of these times
waitTime(W) :-lwaitTime(W), W = #max{D: lwaitTime(D)}.

% Add waiting moves until the non evading robot has freed the path (not necessary)
timeToWait(T0..T0 + W):- chosenStep((X,Y),T0,R,D), waitTime(W).

#show.
#show position(R,(X,Y),T) : position(R,(X,Y),T), chosenStep(_,_,R',_), R != R'.
#show position(R,(X,Y),T) : position(R,(X,Y),T), chosenStep(_,T',R,_), T < T'.
#show position(R,(X,Y),T) : chosenStep((X,Y),T,R,D).
#show position(R,(X,Y),T) : chosenStep((X,Y),T0,R,D), timeToWait(T).
#show position(R,(X',Y'),T0+W + 1) : chosenStep((X,Y),T0,R,D), waitTime(W), position(R,(X',Y'),T0-1).
#show position(R,(X',Y'),T) : chosenStep((X,Y),T0,R,D), waitTime(W), position(R,(X',Y'),T'),T' >= T0, T = T'+W+2.





