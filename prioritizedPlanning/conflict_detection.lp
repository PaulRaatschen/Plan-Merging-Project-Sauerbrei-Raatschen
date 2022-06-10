


% Vertex Conflict
conflict(R) :- rPosition(R,(X,Y),T), rPosition(R',(X,Y),T), R!=R'.

% Edge Conflict
conflict(R) :- rPosition(R,(X,Y),T-1), rPosition(R,(X',Y'),T), rPosition(R',(X',Y'),T-1), rPosition(R',(X,Y),T), R!=R.

% Avoid stationary robots
conflict(R) :- rPosition(R,(X,Y),T), goal(R',(X,Y)), rPosition(R',(X,Y),_), R!=R'.

numOfConflict(R,N) :- N = #count{ conflict(R) }, robot(R).

#show numOfConflict/2.
