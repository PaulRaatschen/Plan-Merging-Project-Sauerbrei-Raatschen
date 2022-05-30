% Takes output from coliisionTRPosition
% Turns the rPosition to aquivalent rPositionX, so that while iterating the old rPositions from previous iterations won't be carried over


conflict(-1,-1,-1,-1,-1,-1). %Placeholder


% Unpack nodes
node((X,Y)) :- init(object(node,_),value(at,(X,Y))).

% Initialize robot position



rPositionX(R,(X,Y),T) :- rPosition(R,(X,Y),T).

% Detect intersecting paths
vertextCollision((X,Y),T,R,R') :- rPositionX(R,(X,Y),T), rPositionX(R',(X,Y),T), R!=R', rPositionX(R,(X0,Y0),T-1), rPositionX(R',(X1,Y1),T-1).


% Detect overlaying paths
edgeCollision((X,Y),(X',Y'),T,R,R',bothmoving) :- rPositionX(R,(X,Y),T), rPositionX(R',(X',Y'),T), rPositionX(R,(X',Y'),T+1), rPositionX(R',(X,Y),T+1), R != R',|X-X'|+|Y-Y'| > 0 , not conflict(_,_,T,R,R',_).



% Detect overlaying paths with intersecting position
edgeCollision((X0,Y0),(X1,Y1),T,R,R',bothmoving) :- rPositionX(R,(X,Y),T), rPositionX(R',(X,Y),T), R!=R', rPositionX(R,(X1,Y1),T+1), rPositionX(R',(X1,Y1),T-1), rPositionX(R,(X0,Y0),T-1), rPositionX(R',(X0,Y0),T+1), R != R',|X0-X1|+|Y0-Y1| > 0, not conflict(_,_,T,R,R',_).
edgeCollision(P,P',T+1,R,R',onewaiting) :- rPositionX(R,P,T), rPositionX(R,P,T+1),rPositionX(R',P',T), rPositionX(R',P,T+1), R != R', P != P', not conflict(_,_,T+1,R,R',_).







#show node/1.
#show rPositionX/3.
#show vertextCollision/4.
#show edgeCollision/6.
#show init/2.
