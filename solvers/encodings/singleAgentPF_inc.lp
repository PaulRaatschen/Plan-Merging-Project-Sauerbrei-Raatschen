#program base.

#program step(t).

{ move(r,D,t) : direction(D) } 1.

position(r,(X+DX,Y+DY),t) :- move(r,(DX,DY),t), position(r,(X,Y),t-1).

:- move(r,(DX,DY),t), position(r,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(r,C,t) :- position(r,C,t-1), not move(r,_,t).

% Vertex Conflict
:- position(r,(X,Y),t), position(R,(X,Y),t), r!=R.

% Edge Conflict
:- position(r,(X,Y),t-1), move(r,(DX,DY),t), position(R,(X+DX,Y+DY),t-1), position(R,(X,Y),t), r!=R.

% Avoid stationary robots
:- position(r,(X,Y),t), goal(R,(X,Y)), goalReached(R,T), r!=R, t > T.

 % CBS Constraints
:- constraint(r,C,t), position(r,C,t).
:- constraint(r,C,M,t-1), position(r,C,t-1), move(r,M,t).
block(1..T-1) :- constraint(r,C,T), goal(r,C).

#program check(t).
#external query(t).

goalReached(r,t) :- goal(r,C), position(r,C,t), not block(t).
:- not goalReached(r,t), query(t).

#show.
#show position(R,C,T) : r = R, position(R,C,T).
#show goalReached(R,T) : r = R, goalReached(R,T).
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).