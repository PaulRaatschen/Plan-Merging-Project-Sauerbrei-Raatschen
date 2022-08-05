rPositionX(-1,-1,-1).

occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPosition(R,(X,Y), T), rPosition(R,(X',Y'), T+1) , DX = X'-X , DY=Y'-Y.
occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPositionX(R,(X,Y), T), rPositionX(R,(X',Y'), T+1) , DX = X'-X , DY=Y'-Y.

#show init/2.
#show occurs/3.