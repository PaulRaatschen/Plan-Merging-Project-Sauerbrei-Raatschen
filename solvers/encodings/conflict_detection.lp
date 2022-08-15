
% Vertex Conflict
conflict(vertex,R,R',(X,Y),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', position(R,(X0,Y0),T-1),position(R',(X0',Y0'),T-1), X0 != X0', Y0 != Y0'.

% Edge Conflict
conflict(edge,R,R',((X,Y),(X',Y')),T-1) :- position(R,(X,Y),T-1), position(R,(X',Y'),T), position(R',(X',Y'),T-1), position(R',(X,Y),T), R!=R'.
conflict(edge,R,R',(P,P'),T-1) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', not conflict(vertex,R,R',(X,Y),T),position(R,P,T-1), position(R',P',T-1).
#show conflict/5.
#show conflict/6.
