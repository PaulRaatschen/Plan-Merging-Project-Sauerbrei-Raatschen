#program base.

#program step(t).

{ move(R,D,t) : direction(D) } 1 :- planning(R).

position(R,(X+DX,Y+DY),t) :- move(R,(DX,DY),t), position(R,(X,Y),t-1).

:- move(R,(DX,DY),t), position(R,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(R,C,t) :- position(R,C,t-1), not move(R,_,t), planning(R).

% Vertex Conflict
:- position(R,(X,Y),t), position(R',(X,Y),t), R!=R', planning(R).

% Edge Conflict
:- position(R,(X,Y),t-1), move(R,(DX,DY),t), position(R',(X+DX,Y+DY),t-1), position(R',(X,Y),t), R!=R', planning(R), planning(R').

% CBS constraints
:- constraint(R,C,t), position(R,C,t).

:- constraint(r,C,M,t-1), position(r,C,t-1), move(r,M,t).

#program check(t).
#external query(t).

cost(N,t) :- N = #sum{1, R, T : position(R,C,T), not goal (R,C), planning(R)}.

#minimize{ N : cost(N,t) }.

goalReached(R,t) :- goal(R,C), position(R,C,t), planning(R).

:- not goalReached(R,t), planning(R), query(t).

#show.
#show planning/1.
#show position/3.
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).