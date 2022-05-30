#include <incmode>.

#include "./setup.lp".

#const r = 1.


#program step(t).

{move(r,D,t) : direction(D)} 1.

rPosition(r,(X+DX,Y+DY),t) :- rPosition(r,(X,Y),t-1), move(r,(DX,DY),t).

rPosition(r,C,t) :- rPosition(r,C,t-1), not move(r,_,t).

:- rPosition(r,(X,Y),_), not node(X,Y).

goalReached(r) :- rPosition(r,(X,Y),t), goal(r,(X,Y)).

:- not goalReached(r).

% Vertex Conflict
:- rPosition(r,(X,Y),t), rPosition(R,(X,Y),t), R!=r.

% Edge Conflict
:- rPosition(r,(X,Y),t), move(r,(DX,DY),t+1), rPosition(R,(X+DX,Y+DY),t), rPosition(R,(X,Y),t+1), R!=r.


#show.
#show rPosition/3.
#show occurs(object(robot,r), action(move,D),T) : move(r,D,T).