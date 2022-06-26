% Takes a plan with contains occurs as well as the inits defining the layout

% Initialize robot position
rPosition(R,(X,Y), 0) :- init(object(robot,R),value(at,(X,Y))).

% Determine robot position at time T based on moves
rPosition(R,(X+DX,Y+DY),T+1) :- occurs(object(robot,R),action(move,(DX,DY)),T+1), rPosition(R,(X,Y),T).

numOfRobots(N) :- N = #count{ R : robot(R)}.

#show rPosition/3.
#show init/2.
#show numOfRobots/1.