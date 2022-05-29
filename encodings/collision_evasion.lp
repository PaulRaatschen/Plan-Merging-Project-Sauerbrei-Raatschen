% Takes output of find_conflict.lp as input

earliestEdgeCollision(TMIN) :- TMIN = #min{T : edgeCollision(_,_,T,_,_,_)}.

{firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S): edgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S), earliestEdgeCollision(T0)} = 1.
firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',S) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R',R,S).

conflict(A,B,C,D,E,F) :- firstedgeCollision(A,B,C,D,E,F).

% Find an evasion step for one of the robots that frees the path for the other one
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',onewaiting), not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|+2, node((X,Y)). %<= 1 zu =1 falls Probleme
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',onewaiting), not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|+1, node((X,Y)). %<= 1 zu =1 falls Probleme
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',onewaiting), not rPositionX(R',(X,Y),_), rPosition(R,P,T+2),rPosition(R',P,T+1), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|+1, node((X,Y)). %<= 1 zu =1 falls Probleme
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',bothmoving),R != R', not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|, node((X,Y)). %<= 1 zu =1 falls Probleme
evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',bothmoving),R != R', not rPositionX(R',(X,Y),_), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 0, D = |T0-T|-6, node((X,Y)). %<= 1 zu =1 falls Probleme

%evasionStep((X,Y),T+1,R,D) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R'), rBacktraking(R,(X',Y'),T), |X-X'|+|Y-Y'| = 1, D = |T0-T|, node((X,Y)).


% Problem: Rule does not allow for evasion step to be on other robots path, though there could be valid evasion steps there  (same problem as in line 10)



% Initialize the backtraking from the collision point 
rBacktraking(R,(X,Y),T0-1) :- firstedgeCollision((X0,Y0),(X0',Y0'),T0,R,R',_), rPositionX(R,(X,Y),T0-1).

% Backtrack along the path (ideally only until an evasion step is found)
rBacktraking(R,(X,Y),T) :- rBacktraking(R,(X',Y'),T+1), rPositionX(R,(X,Y),T). % not evasion_step(_,T0,_,_) with T0 < T (dont know how to do in clingo)

% Determine the evasion step with the smallest distance to the collision point in order to minimize waiting time
minDistance(DMAX) :- DMAX = #min{D : evasionStep((X,Y),T,R,D)}. 

% Choose one evasion step with minimal distance to the collision point
{chosenStep((X,Y),T,R,D): evasionStep((X,Y),T,R,D)} = 1 :- minDistance(D). 

% Time that evading robot has to wait
lwaitTime(W) :- chosenStep((X,Y),T0,R,D), rPositionX(R,(X',Y'),T0-1), R != R' ,rPositionX(R',(X',Y'),TD), W = TD - T0,firstedgeCollision(_,_,_,R,R',bothmoving).
lwaitTime(W) :- chosenStep((X,Y),T0,R,D),  R != R' ,rPositionX(R',(X',Y'),TD),|X-X'|+|Y-Y'| == 1, W = TD - T0,firstedgeCollision(_,_,_,R,R',bothmoving).
lwaitTime(W) :- firstedgeCollision((X0,Y0),_,_,R,R',onewaiting),chosenStep((X,Y),T0,R,D), rPositionX(R',(X0,Y0),T1),rPositionX(R',(X1,Y1),T2), |X0-X1|+|Y0-Y1| = 1, T1 < T2, W = T2-T0-1 .

waitTime(W) :-lwaitTime(W), W = #max{D: lwaitTime(D)}.
%waitTime(W) :- chosenStep((X,Y),T0,R,0), rPositionX(R,(X',Y'),T0-1),rPositionX(R',(X',Y'),TD), W = TD- (T0-1).


% Translate back to occurs and init syntax. Incorporate evasion step into path of chosen robot
%init(object(robot,R),value(at,(X,Y))) :- rPositionX(R,(X,Y), 0).

% Keep the path of the non evading robot unaltered
%occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPositionX(R,(X',Y'),T+1), rPositionX(R,(X,Y),T), chosenStep(_,_,R',_),firstedgeCollision(_,_,_,R',R), not R = R', DX = X'-X, DY = Y'-Y.
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y),T), chosenStep(_,_,R',_),firstedgeCollision(_,_,_,R',R,_), not R = R'.

% Keep the path of the evading robot unaltered until evasion step
%occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPositionX(R,(X',Y'),T+1), rPositionX(R,(X,Y),T), chosenStep(_,T',R,_), T+1 < T', DX = X'-X, DY = Y'-Y,firstedgeCollision(_,_,_,R',R). 
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y),T), chosenStep(_,T',R,_), T < T'.

% Add evasion step to the path
%occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPositionX(R,(X,Y),T), chosenStep((X',Y'),T+1,R,D), DX = X' - X, DY = Y' - Y,firstedgeCollision(_,_,_,R',R).
rPosition(R,(X,Y),T) :- chosenStep((X,Y),T,R,D).

% Initialize waiting moves if evasion step is not made at the collision point
%occurs(object(robot,R),action(move,(0,0)),T0+1) :- chosenStep((X,Y),T0,R,D), D > 0,firstedgeCollision(_,_,_,R',R).
%rPosition(R,(X,Y),T+1) :- chosenStep((X,Y),T,R,D), D>0.

% Add waiting moves until the non evading robot has freed the path (not necessary)
%occurs(object(robot,R),action(move,(0,0)),T+1) :- occurs(object(robot,R),action(move,(0,0)),T), chosenStep((X,Y),T0,R,D), waitTime(W), T < T0 + W,firstedgeCollision(_,_,_,R',R).
timeToWait(T0..T0 + W):- chosenStep((X,Y),T0,R,D), waitTime(W).
%{rPosition(R,(X,Y),T): chosenStep((X,Y),T0,R,D), timeToWait(T)} = W+1 :- waitTime(W).
rPosition(R,(X,Y),T):- chosenStep((X,Y),T0,R,D), timeToWait(T).
% Move evading robot back onto original path 
%occurs(object(robot,R),action(move,(DX,DY)),T0 + W + 1) :- chosenStep((X,Y),T0,R,D), rPositionX(R,(X',Y'),T0-1),firstedgeCollision(_,_,_,R,R'), DX = X'-X, DY = Y'-Y, waitTime(W).
rPosition(R,(X',Y'),T0+W + 1):- chosenStep((X,Y),T0,R,D), waitTime(W), rPositionX(R,(X',Y'),T0-1).

% Continue original path with delay
%occurs(object(robot,R),action(move,(DX,DY)),T + 1 + W + 1) :- rPositionX(R,(X',Y'),T), rPositionX(R,(X,Y),T-1), chosenStep(_,T0,R,D), T >= T0, DX = X'- X, DY = Y' - Y, waitTime(W),firstedgeCollision(_,_,_,R',R).
rPosition(R,(X',Y'),T):- chosenStep((X,Y),T0,R,D), waitTime(W), rPositionX(R,(X',Y'),T'),T' >= T0, T = T'+W+2.

%occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPositionX(R,(X,Y), T),rPositionX(R,(X',Y'), T+1),DX = X'-X,DY=Y'-Y, firstedgeCollision(_,_,_,R1,R2), R != R1, R != R2.
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y), T), firstedgeCollision(_,_,_,R1,R2,_), R != R1, R != R2.


#show init/2.
#show waitTime/1.
#show rPosition/3.
#show conflict/6.






