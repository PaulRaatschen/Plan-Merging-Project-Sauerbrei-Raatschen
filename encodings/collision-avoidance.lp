maxTime(2*M) :- M = #max{T : rPosition(_,_,T)}.

timeStep(T,R) :- T = 0..M, maxTime(M), rPosition(R,_,_).



{newRPosition(R,(X,Y),T) : timeStep(T,R) } = 1 :- rPosition(R,(X,Y),T').

:- newRPosition(R,P,T), rPosition(R,P,T'), T < T'.

:- newRPosition(R,P',T') ,newRPosition(R,P,T), rPosition(R,P',T0), rPosition(R,P,T1), T0 < T1, T' < T. 

:- newRPosition(R,P,T), newRPosition(R',P,T), R' != R.

:- newRPosition(R,P,T), newRPosition(R, P', T), P != P'.

occurs(object(robot,R),action(move,(DX,DY)),T) :- newRPosition(R,(X,Y),T), newRPosition(R,(X',Y'),T+1), DX = X-X', DY = Y-Y'.

%#show newRPosition/3.
#show occurs/3.
%#show init/2.