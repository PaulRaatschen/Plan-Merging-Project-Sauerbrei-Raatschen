% Takes output of find_conflict.lp as input

% Determines ealiest conflict
earliestEdgeConflict(TMIN) :- TMIN = #min{T : conflict(edge,_,_,_,T)}.



{firstEdgeConflict((X0,Y0),(X0',Y0'),T0,R,R'): conflict(edge,R,R',((X0,Y0),(X0',Y0')),T0), earliestEdgeConflict(T0)} = 1.
%firstEdgeConflict(P,P',T0,R,R') :- firstEdgeConflict(P',P,T0,R',R).

edgeConflict(P,P',T,R,R') :- firstEdgeConflict(P,P',T,R,R').
edgeConflict(P,P',T,R,R') :- firstEdgeConflict(P',P,T,R',R).

distance(|X1-X2| + |Y1-Y2|):- edgeConflict((X1,Y1),(X2,Y2),_,_,_).

% Initialize the backtracking from the conflict point for both agents
%backtracking(R,(X,Y),T0,T0+DIS) :- edgeConflict(_,_,T0,R,R'), position(R,(X,Y),T0), distance(DIS).
% Backtrack along the path of the other agent (ideally only until an evasion step is found)
%backtracking(R,(X,Y),T-1,TD+1) :- backtracking(R,(X',Y'),T,TD),position(R',(X',Y'),TD), position(R',(X,Y),TD+1),edgeConflict(_,_,_,R,R'), position(R',_,TD+2). % not evasion_step(_,T0,_,_) with T0 < T (dont know how to do in clingo)

%backtracking(R,(X,Y),T-1,TD+1) :- backtracking(R,(X',Y'),T,TD),position(R,(X',Y'),T), position(R,(X,Y),T-1).

% Find an evasion step for one of the robots that frees the path for the other one

%evasionStep((X,Y),T0-D,R,D) :- edgeConflict(_,_,T0,R,R'), not position(R',(X,Y),_), backtracking(R,(X',Y'),T,TD),  D = |TD|, node((X,Y)),|X-X'|+|Y-Y'| <= 1. 
evasionStep((X,Y),T+1,R,T0-T+DIS) :- edgeConflict(_,_,T0,R,R'), not position(R',(X,Y),_), position(R,(X',Y'),T), T <= T0, node((X,Y)),|X-X'|+|Y-Y'| <= 1, distance(DIS).
evasionStep((X,Y),T+-T0,R,T+DIS) :- edgeConflict(_,_,T0,R,R'), not position(R',(X,Y),_), position(R',(X',Y'),T), T > T0, node((X,Y)),|X-X'|+|Y-Y'| <= 1,distance(DIS). 

% Determine the evasion step with the smallest distance to the conflict point in order to minimize waiting time
minDistance(DMAX) :- DMAX = #min{D : evasionStep((X,Y),T,R,D), not position(R',(X,Y),T), position(R',_,_)}. 

% Choose one evasion step with minimal distance to the conflict point
{chosenStep((X,Y),T,R,D): evasionStep((X,Y),T,R,D)} = 1 :- minDistance(D). 

% Time that evading robot has to wait, either when he passes the point where the robot evaded, or when he passes the point where he had to stop
%lwaitTime(W) :- chosenStep((X,Y),T0,R,D), position(R,(X',Y'),T0-1), R != R' ,position(R',(X',Y'),TD), W = TD - T0,firstEdgeConflict(_,_,_,R,R').
%lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0+1), W = TD - T0,firstEdgeConflict(_,_,_,R,R').
%lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,position(R',(X',Y'),TD),position(R,(X',Y'),T0), W = TD - T0,firstEdgeConflict(_,_,_,R,R').
lwaitTime(D) :- chosenStep((X,Y),T0,R,D).
lwaitTime(0).

% Longest of these times
waitTime(W) :-lwaitTime(W), W = #max{D: lwaitTime(D)}.

% Add waiting moves until the non evading robot has freed the path (not necessary)
timeToWait(T0..T0 + W):- chosenStep((X,Y),T0,R,D), waitTime(W).
timeToMoveBack(1..|T|):- chosenStep(_,T,_,_), T<0.

#show.
% Paths of non evading agents and uninvolved agents
#show position(R,(X,Y),T) : position(R,(X,Y),T), chosenStep(_,_,R',_), R != R'.

% If agent has to move off his own path if he needs to follow other agent for an sidestep
    #show position(R,(X,Y),T): chosenStep(_,T',R,_),T'<0,position(R',P,T),position(R,P,0),position(R',(X,Y),T+MoveBack), timeToMoveBack(MoveBack).
    #show position(R,(X,Y),|T'|): chosenStep((X,Y),T',R,D),T'<0.
    #show position(R,(X,Y),|T'|+MoveBack+W): chosenStep(_,T',R,_),T'<0,position(R',P,T),position(R,P,0),position(R',(X,Y),T+T'-MoveBack), timeToMoveBack(MoveBack),waitTime(W).
    #show position(R,P,|T'|+MoveBack+W+DT): chosenStep(_,T',R,_),T'<0,position(R,P,DT), timeToMoveBack(MoveBack),waitTime(W).
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








