

maxTime(M + 1) :- M = #max{T : rPosition(_,_,T)}.

timeStep(T) :- T = 0..M, maxTime(M).

{newRPosition(R,(X,Y),T) : timeStep(T)} = 1 :- rPosition(R,(X,Y),T').

:- newRPosition(R,P,T), rPosition(R,P,T'), T < T'.

:- newRPosition(R,P',T') ,newRPosition(R,P,T), rPosition(R,P',T0), rPosition(R,P,T1), T1 > T0, T' < T. 

:- newRPosition(R,P,T), newRPosition(R',P,T), R' != R.


occurs(object(robot,R),action(move,(DX,DY)),T) :- newRPosition(R,(X,Y),T), newRPosition(R,(X',Y'),T+1), DX = X'-X, DY = Y'-Y.

#show newRPosition/3.
#show occurs/3.
#show init/2.