#program base.

%#include <incmode>.
%#include "setup.lp".

#program step(t).
%#external constraint(R,C,T) : robot(R), node(C), T = 1..t.


{ move(r,D,t) : direction(D) } 1.

position(r,(X+DX,Y+DY),t) :- move(r,(DX,DY),t), position(r,(X,Y),t-1).

:- move(r,(DX,DY),t), position(r,(X,Y) ,t-1), not node((X+DX,Y+DY)).

position(r,C,t) :- position(r,C,t-1), not move(r,_,t).

% Vertex Conflict
:- position(r,(X,Y),t), position(R,(X,Y),t), r!=R.

% Edge Conflict
:- position(r,(X,Y),t-1), move(r,(DX,DY),t), position(R,(X+DX,Y+DY),t-1), position(R,(X,Y),t), r!=R.

% Avoid stationary robots
:- position(r,(X,Y),t), goal(R,(X,Y)), position(R,(X,Y),T), r!=R, t >= T.
:- position(r,(X,Y),t), goal(r,(X,Y)), position(R,(X,Y),T), r!=R, t <= T.

:- constraint(r,C,t), position(r,C,t).

:- constraint(r,C,M,t-1), position(r,C,t-1), move(r,M,t).

#program check(t).
#external query(t).

goalReached(r,t) :- goal(r,C), position(r,C,t).

:- not goalReached(r,t), query(t).

#show.
#show position/3.
#show goalReached/2.
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).