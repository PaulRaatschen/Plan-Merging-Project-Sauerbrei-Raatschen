occurs(object(robot,R),action(move,(DX,DY)),T) :- newRPosition(R,(X,Y),T,I), newRPosition(R,(X',Y'),T+1,I), DX = X'-X, DY = Y'-Y.

#show occurs/3.