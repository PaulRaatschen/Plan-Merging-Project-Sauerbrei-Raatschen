#program step(t).

{move(r,D,t) : direction(D)} 1.

:- { rPosition(r,C,t) } != 1.

rPosition(r,(X+DX,Y+DY),t) :- rPosition(r,(X,Y),t-1), move(r,(DX,DY),t).

rPosition(r,C,t) :- rPosition(r,C,t-1), not move(r,_,t).

:- rPosition(r,(X,Y),_), not node(X,Y).

#program check(t).

:- not goal(r,(X,Y)), rPosition(r,(X,Y),t), query(t).

#show rPosition/3.
