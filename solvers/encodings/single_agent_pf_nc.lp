#program base.

#program step(t).

{ move(r,D,t) : direction(D) } 1.

position(r,(X+DX,Y+DY),t) :- move(r,(DX,DY),t), position(r,(X,Y),t-1).

:- move(r,(DX,DY),t), position(r,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(r,C,t) :- position(r,C,t-1), not move(r,_,t).

#program check(t).
#external query(t).

goalReached(r,t) :- goal(r,C), position(r,C,t).
:- not goalReached(r,t), query(t).

#show.
#show position(R,C,T) : r = R, position(R,C,T).
#show goalReached(R,T) : r = R, goalReached(R,T).
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).