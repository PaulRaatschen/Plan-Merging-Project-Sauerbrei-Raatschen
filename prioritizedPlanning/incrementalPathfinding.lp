%#include <incmode>.

%#include "setup.lp".
%#const imin = 12.

#program step(t).

{move(r,D,t) : direction(D)} 1.

:- { rPosition(r,C,t) } != 1.

rPosition(r,(X+DX,Y+DY),t) :- rPosition(r,(X,Y),t-1), move(r,(DX,DY),t).

rPosition(r,C,t) :- rPosition(r,C,t-1), not move(r,_,t).

:- rPosition(r,(X,Y),_), not node(X,Y).

% Vertex Conflict
:- rPosition(r,(X,Y),t), rPosition(R,(X,Y),t), R!=r.

% Edge Conflict
:- rPosition(r,(X,Y),t-1), move(r,(DX,DY),t), rPosition(R,(X+DX,Y+DY),t-1), rPosition(R,(X,Y),t), R!=r.

% Avoid stationary robots
:- rPosition(r,(X,Y),t), goal(R,(X,Y)), rPosition(R,(X,Y),T),t >=T, r!=R.
:- rPosition(r,(X,Y),t), goal(r,(X,Y)), rPosition(R,(X,Y),T),t <=T, r!=R.

#program check(t).

:- not goal(r,(X,Y)), rPosition(r,(X,Y),t), query(t).

#show.
#show rPosition/3.
#show occurs(object(robot,r), action(move,D),T) : move(r,D,T).