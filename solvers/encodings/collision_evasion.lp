% Takes output of find_conflict.lp as input

% Determines ealiest collision
earliestEdgeCollision(TMIN) :- TMIN = #min{T : edgeCollision(_,_,T,_,_,_)}.

{firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S): edgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S), earliestEdgeCollision(T0)} = 1.
firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R',R,S).

% Indipendenten object used for showing output
conflict(A,B,C,D,E,F) :- firstedgeCollision(A,B,C,D,E,F).

% Find an evasion step for one of the robots that frees the path for the other one

evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',_),R != R', not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|+1, node((X,Y)).
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',_),R != R', not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|, node((X,Y)). 


% Initialize the backtraking from the collision point 
rBacktraking(R,(X,Y),T0-1) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',_), rPositionX(R,(X,Y),T0-1).

% Backtrack along the path (ideally only until an evasion step is found)
rBacktraking(R,(X,Y),T) :- rBacktraking(R,(X',Y'),T+1), rPositionX(R,(X,Y),T). % not evasion_step(_,T0,_,_) with T0 < T (dont know how to do in clingo)

% Determine the evasion step with the smallest distance to the collision point in order to minimize waiting time
minDistance(DMAX) :- DMAX = #min{D : evasionStep((X,Y),T,R,D)}. 

% Choose one evasion step with minimal distance to the collision point
{chosenStep((X,Y),T,R,D): evasionStep((X,Y),T,R,D)} = 1 :- minDistance(D). 

% Time that evading robot has to wait, either when he passes the point where the robot evaded, or when he passes the point where he had to stop
lwaitTime(W) :- chosenStep((X,Y),T0,R,D), rPositionX(R,(X',Y'),T0-1), R != R' ,rPositionX(R',(X',Y'),TD), W = TD - T0,firstedgeCollision(_,_,_,R,R',_).
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,rPositionX(R',(X',Y'),TD),rPositionX(R,(X',Y'),T0+1), W = TD - T0,firstedgeCollision(_,_,_,R,R',_).
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,rPositionX(R',(X',Y'),TD),rPositionX(R,(X',Y'),T0), W = TD - T0,firstedgeCollision(_,_,_,R,R',_).
lwaitTime(0).

% Longest of these times
waitTime(W) :-lwaitTime(W), W = #max{D: lwaitTime(D)}.

% Keep the path of the non evading robots unaltered
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y),T), chosenStep(_,_,R',_), R != R'.

% Keep the path of the evading robot unaltered until evasion step
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y),T), chosenStep(_,T',R,_), T < T'.

% Add evasion step to the path
rPosition(R,(X,Y),T) :- chosenStep((X,Y),T,R,D).

% Add waiting moves until the non evading robot has freed the path (not necessary)

timeToWait(T0..T0 + W):- chosenStep((X,Y),T0,R,D), waitTime(W).
rPosition(R,(X,Y),T):- chosenStep((X,Y),T0,R,D), timeToWait(T).

% Move evading robot back onto original path 

rPosition(R,(X',Y'),T0+W + 1):- chosenStep((X,Y),T0,R,D), waitTime(W), rPositionX(R,(X',Y'),T0-1).

% Continue original path with delay
rPosition(R,(X',Y'),T):- chosenStep((X,Y),T0,R,D), waitTime(W), rPositionX(R,(X',Y'),T'),T' >= T0, T = T'+W+2.



#show init/2.
#show waitTime/1.
#show rPosition/3.
#show conflict/6.






