#program base.

#program step(t).

{ move(R,D,t) : direction(D) } 1 :- planning(R).

:- position(R,C,T), goalReached(R',T'), goal(R',C), T' < T.

position(R,(X+DX,Y+DY),t) :- move(R,(DX,DY),t), position(R,(X,Y),t-1).

:- move(R,(DX,DY),t), position(R,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(R,C,t) :- position(R,C,t-1), not move(R,_,t), planning(R), not goalReached(R,_).

% Vertex Conflict
:- position(R,(X,Y),t), position(R',(X,Y),t), R!=R', planning(R).

% Edge Conflict
:- position(R,(X,Y),t-1), move(R,(DX,DY),t), position(R',(X+DX,Y+DY),t-1), position(R',(X,Y),t), R!=R', planning(R), planning(R').

% CBS constraints
:- constraint(R,C,t), position(R,C,t), planning(R).
:- constraint(R,C,M,t-1), position(R,C,t-1), move(R,M,t), planning(R).
block(1..T-1) :- constraint(R,C,T), goal(R,C), planning(R).
:- meta_constraint(A,C,t), position(R,C,t).
:- meta_constraint(A,C,M,t-1), position(R,C,t-1), move(R,M,t).
block(1..T-1) :- meta_constraint(A,C,T), goal(R,C), planning(R).

#program check(t).
#external query(t).

cost(N,t) :- N = #sum{1, R, T : position(R,C,T), not goal (R,C), planning(R)}.

#minimize{ N : cost(N,t)}.

goalReached(R,t) :- goal(R,C), position(R,C,t), planning(R), not block(t).

:- not goalReached(R,_), planning(R), query(t).

#show.
#show goalReached(R,T) : planning(R), goalReached(R,T).
#show position(R,C,T) : planning(R), position(R,C,T).
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).