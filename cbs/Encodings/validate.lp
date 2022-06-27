% Vertex Conflict
conflict(vertex,R,R',T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R'.

% Edge Conflict
conflict(edge,R,R',T) :- position(R,(X,Y),T-1), move(R,(DX,DY),T), position(R',(X+DX,Y+DY),T-1), position(R,(X,Y),T), R!=R'.

minConflict(K,R,R',T') :- conflict(K,R,R',T'), T' = #min{ T : conflict(_,_,_,T)}. 

#show conflict/4.