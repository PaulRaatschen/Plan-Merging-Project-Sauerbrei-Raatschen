occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPosition(R,(X,Y), T), rPosition(R,(X',Y'), T+1) , DX = X'-X , DY=Y'-Y.

#show init/2.
#show occurs/3.