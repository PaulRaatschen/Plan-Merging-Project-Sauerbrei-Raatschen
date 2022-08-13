% Vertex Conflict
conflict(vertex,R,R',(X,Y),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R'.

% Edge Conflict
conflict(edge,R,R',((X,Y),(X',Y')),T-1) :- position(R,(X,Y),T-1), position(R,(X',Y'),T), position(R',(X',Y'),T-1), position(R',(X,Y),T), R!=R'.

% Goal conflict
conflict(edge,R,R',(X,Y),T) :- position(R,(X,Y),T), goal(R',(X,Y)), goalReached(R',T'), R!=R', T >= T'.

% First conflict that occurs
minConflict(K,R,R',C,T') :- conflict(K,R,R',C,T'), T' = #min{ T : conflict(_,_,_,_,T)}. 

#show minConflict/5.