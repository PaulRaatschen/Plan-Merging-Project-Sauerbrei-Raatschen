#program step(t).

{move(r,D,t) : direction(D)} 1.

:- { position(r,C,t) } != 1.

position(r,(X+DX,Y+DY),t) :- position(r,(X,Y),t-1), move(r,(DX,DY),t).

position(r,C,t) :- position(r,C,t-1), not move(r,_,t).

:- position(r,(X,Y),_), not node((X,Y)).

#program check(t).

:- not goal(r,(X,Y)), position(r,(X,Y),t), query(t).

#show position/3.
