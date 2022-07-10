%#include <incmode>.
%#include "setup.lp".

#program step(t).

{move(r,D,t) : direction(D)} 1.

:- { position(r,C,t) } != 1.

position(r,(X+DX,Y+DY),t) :- position(r,(X,Y),t-1), move(r,(DX,DY),t).

position(r,C,t) :- position(r,C,t-1), not move(r,_,t).

:- position(r,(X,Y),_), not node((X,Y)).

% Vertex Conflict
:- position(r,(X,Y),t), position(R,(X,Y),t), R!=r.

% Edge Conflict
:- position(r,(X,Y),t-1), move(r,(DX,DY),t), position(R,(X+DX,Y+DY),t-1), position(R,(X,Y),t), R!=r.

% Avoid robots in their goal and passing over r when it is done
:- position(r,(X,Y),t), goal(R,(X,Y)), position(R,(X,Y),T),t >=T, r!=R.

#program check(t).

:- not goal(r,(X,Y)), position(r,(X,Y),t), query(t).

#show.
#show position/3.
#show occurs(object(robot,r), action(move,D),T) : move(r,D,T).