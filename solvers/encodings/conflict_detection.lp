
% Vertex Conflict
conflict(vertex,R,R',(X,Y),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', position(R,(X0,Y0),T-1),position(R',(X0',Y0'),T-1).

% Edge Conflict
%Both agents switch position
conflict(edge,R,R',((X,Y),(X',Y')),T) :-  position(R,(X,Y),T), position(R',(X',Y'),T), position(R,(X',Y'),T+1), position(R',(X,Y),T+1), R != R',|X-X'|+|Y-Y'| > 0.
%Both agent go onto the same spot
conflict(edge,R,R',((X0,Y0),(X1,Y1)),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', position(R,(X1,Y1),T+1), position(R',(X1,Y1),T-1), position(R,(X0,Y0),T-1), position(R',(X0,Y0),T+1), R != R',|X0-X1|+|Y0-Y1| > 0.
%One agent waits while the other goes into him
conflict(edge,R,R',(P,P'),T+1):- position(R,P,T), position(R,P,T+1),position(R',P',T), position(R',P,T+1), R != R', P != P'.
#show conflict/5.
#show conflict/6.
