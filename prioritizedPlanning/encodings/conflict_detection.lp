
% Vertex Conflict
conflict(R) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R'.

% Edge Conflict
conflict(R) :- position(R,(X,Y),T-1), position(R,(X',Y'),T), position(R',(X',Y'),T-1), position(R',(X,Y),T), R!=R.

% Avoid stationary robots
conflict(R) :- position(R,(X,Y),T), goal(R',(X,Y)), position(R',(X,Y),_), R!=R'.

numOfConflicts(R,N) :- N = #count{ conflict(R) }, robot(R).

#show numOfConflicts/2.
