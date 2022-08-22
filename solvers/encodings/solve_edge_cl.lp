% Takes output of find_conflict.lp as input

% Determines ealiest conflict
earliestEdgeConflict(TMIN) :- TMIN = #min{T : conflict(edge,_,_,_,T)}.



{firstEdgeConflict((X0,Y0),(X0',Y0'),T0,R,R'): conflict(edge,R,R',((X0,Y0),(X0',Y0')),T0), earliestEdgeConflict(T0)} = 1.
%firstEdgeConflict(P,P',T0,R,R') :- firstEdgeConflict(P',P,T0,R',R).

edgeConflict(P,P',T,R,R') :- firstEdgeConflict(P,P',T,R,R').
edgeConflict(P,P',T,R,R') :- firstEdgeConflict(P',P,T,R',R).

iswaiting(R):- edge(P,P',T,R,R'), position(R,P,T), position(R,P,T+1).

% Initialize the backtraking from the collision point 
rBacktraking(R,(X,Y),T0-1) :- edgeConflict((X0,Y0),(X0',Y0'),T0,R,R'), position(R,(X,Y),T0-1).

% Backtrack along the path (ideally only until an evasion step is found)
rBacktraking(R,(X,Y),T) :- rBacktraking(R,(X',Y'),T+1), position(R,(X,Y),T). % not evasion_step(_,T0,_,_) with T0 < T (dont know how to do in clingo)

% Find an evasion step for one of the robots that frees the path for the other one

evasionStep((X,Y),T+1,R,D) :- edgeConflict((X0,Y0),(X0',Y0'),T0,R,R'),R != R', not position(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|+1, node((X,Y)).
evasionStep((X,Y),T+1,R,D) :- edgeConflict((X0,Y0),(X0',Y0'),T0,R,R'),R != R', not position(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|, node((X,Y)). 




% Determine the evasion step with the smallest distance to the collision point in order to minimize waiting time
minDistance(DMAX) :- DMAX = #min{D : evasionStep((X,Y),T,R,D)}. 

% Choose one evasion step with minimal distance to the collision point
{chosenStep((X,Y),T,R,D): evasionStep((X,Y),T,R,D)} = 1 :- minDistance(D). 

% Time that evading robot has to wait, either when he passes the point where the robot evaded, or when he passes the point where he had to stop
% Wait until the non evading agent past the space where the evading agent evaded
lwaitTime(W) :- chosenStep((X,Y),T0,R,D), position(R,(X',Y'),T0-1), R != R' ,position(R',(X',Y'),TD), W = TD - T0,edgeConflict(_,_,_,R,R'), not iswaiting(R).
% Wait until the non evading agent is on the last time on the spot where the evading agent would be one step after not evading
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0+1), W = TD - T0,edgeConflict(_,_,_,R,R'), not iswaiting(R).
% Wait until the non evading agent is on the spot where the evading agent would be if not evading
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0), W = TD - T0,edgeConflict(_,_,_,R,R'),iswaiting(R).
lwaitTime(0).

% Longest of these times
waitTime(W) :-lwaitTime(W), W = #max{D: lwaitTime(D)}.


% Add waiting moves until the non evading robot has freed the path (not necessary)
timeToWait(T0..T0 + W):- chosenStep((X,Y),T0,R,D), waitTime(W).
timeToMoveBack(1..|T|):- chosenStep(_,T,_,_), T<0.

#show.
% Paths of non evading agents and uninvolved agents
#show position(R,(X,Y),T) : position(R,(X,Y),T), chosenStep(_,_,R',_), R != R'.

% Keep path before evading
#show position(R,(X,Y),T) : position(R,(X,Y),T), chosenStep(_,T',R,_), T < T', T'>=0.
% Do the evasion step
#show position(R,(X,Y),T) : chosenStep((X,Y),T,R,D).
% Wait in the choosen position
#show position(R,(X,Y),T) : chosenStep((X,Y),T0,R,D), timeToWait(T).
% Move the agent back onto his old path
#show position(R,(X',Y'),T0+W + 1) : chosenStep((X,Y),T0,R,D), waitTime(W), position(R,(X',Y'),T0-1), T0 >=0.
% Continue his path
#show position(R,(X',Y'),T) : chosenStep((X,Y),T0,R,D), waitTime(W), position(R,(X',Y'),T'),T' >= T0, T = T'+W+2,T0 >=0.








