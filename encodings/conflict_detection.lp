% Takes instance and plans as input

% Unpack nodes
node((X,Y)) :- init(object(node,_),value(at,(X,Y))).

% Initialize robot position
rPosition(R,(X,Y), 0) :- init(object(robot,R),value(at,(X,Y))).

% Determine robot position at time T based on moves
rPosition(R,(X+DX,Y+DY),T+1) :- occurs(object(robot,R),action(move,(DX,DY)),T+1), rPosition(R,(X,Y),T).

% Detect intersecting paths
vertextCollision((X,Y),T,R,R') :- rPosition(R,(X,Y),T), rPosition(R',(X,Y),T), R!=R', rPosition(R,(X0,Y0),T-1), rPosition(R',(X1,Y1),T-1), not edgeCollision((X0,Y0),(X1,Y1),T,R,R').

% Detect overlaying paths
edgeCollision((X,Y),(X',Y'),T,R,R') :- rPosition(R,(X,Y),T), rPosition(R',(X',Y'),T), rPosition(R,(X',Y'),T+1), rPosition(R',(X,Y),T+1), R != R'.

% Detect overlaying paths with intersecting position
edgeCollision((X0,Y0),(X1,Y1),T,R,R') :- rPosition(R,(X,Y),T), rPosition(R',(X,Y),T), R!=R', rPosition(R,(X1,Y1),T+1), rPosition(R',(X1,Y1),T-1), rPosition(R,(X0,Y0),T-1), rPosition(R',(X0,Y0),T+1), R != R'.


#show node/1.
#show rPosition/3.
#show vertextCollision/4.
#show edgeCollision/5.
#show init/2.
