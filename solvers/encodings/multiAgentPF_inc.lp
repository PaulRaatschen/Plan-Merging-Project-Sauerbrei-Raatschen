#program base.

#include <incmode>.
#include "setup.lp".

#const imax = 50.

#external planning(R) : robot(R).

planning(1..4).

#program step(t).

{ move(R,D,t) : direction(D) } 1 :- planning(R).

position(R,(X+DX,Y+DY),t) :- move(R,(DX,DY),t), position(R,(X,Y),t-1).

:- move(R,(DX,DY),t), position(R,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(R,C,t) :- position(R,C,t-1), not move(R,_,t).

% Vertex Conflict
:- position(R,(X,Y),t), position(R',(X,Y),t), R!=R'.

% Edge Conflict
:- position(R,(X,Y),t-1), move(R,(DX,DY),t), position(R',(X+DX,Y+DY),t-1), position(R',(X,Y),t), R!=R'.

cost(N,t) :- N = #sum{1, R, T: position(R,C,T), not goal (R,C), planning(R)}.

#program check(t).

#minimize{ N : cost(N,t) }.

goalReached(R,t) :- goal(R,C), position(R,C,t).

:- not goalReached(R,_), planning(R), query(t).

#show.
#show cost/2.
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).