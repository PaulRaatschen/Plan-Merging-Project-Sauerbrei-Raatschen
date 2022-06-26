
% Setup

robot(R) :- init(object(robot,R),_).

node(X,Y) :- init(object(node,_),value(at,(X,Y))).

goal(R,(X,Y)) :- init(object(shelf,R),value(at,(X,Y))), robot(R). 

rPosition(R,(X,Y),0) :- init(object(robot,R),value(at,(X,Y))).

direction((X,Y)) :- X=-1..1, Y=-1..1, |X+Y|=1.

numOfRobots(N) :- N = #count{ R : robot(R)}.





