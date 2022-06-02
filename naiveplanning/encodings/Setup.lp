robot(R) :- init(object(robot,R),_).

time(1..horizon).

rPosition(R,(X,Y), 0,0) :- init(object(robot,R),value(at,(X,Y))).

rPosition(R,(X+DX,Y+DY),T+1,0) :- occurs(object(robot,R),action(move,(DX,DY)),T+1), rPosition(R,(X,Y),T).

goal((X,Y),R) :- rPosition(R,(X,Y),horizon).

direction((X,Y)) :- X=-1..1, Y=-1..1, |X+Y|=1.

position((X,Y)):- init(object(node,_),value(at,(X,Y))).

conflict((vertex,(X,Y)),T,R,R',0) :- rPosition(R,(X,Y),T), rPosition(R',(X,Y),T), R!=R'.

conflict((edge,(X,Y),(X',Y')),T,R,R',0) :- rPosition(R,(X,Y),T), rPosition(R',(X',Y'),T), rPosition(R,(X',Y'),T+1), rPosition(R',(X,Y),T+1).









