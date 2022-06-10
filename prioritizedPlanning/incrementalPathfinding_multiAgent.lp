#include <incmode>.

#include "setup.lp".
%#const imin = 12.

#program step(t).

{move(R,D,t) : direction(D)} 1 :- robot(R).

rPosition(R,(X+DX,Y+DY),t) :- rPosition(R,(X,Y),t-1), move(R,(DX,DY),t), robot(R).

rPosition(R,C,t) :- rPosition(R,C,t-1), not move(R,_,t), robot(R).
rPosition(R,C,t) :- rPosition(R,C,t-1), not move(R,_,t), robot(R).

:- rPosition(R,(X,Y),_), not node(X,Y), robot(R).

% Vertex Conflict
conflict(R,R',t) :- rPosition(R,(X,Y),t), rPosition(R',(X,Y),t), R!=R.

% Edge Conflict
conflict(R,R',t) :- rPosition(R,(X,Y),t-1), move(R,(DX,DY),t), rPosition(R',(X+DX,Y+DY),t-1), rPosition(R',(X,Y),t), R!=R.

% Avoid stationary robots
conflict(R,R',t) :- rPosition(R,(X,Y),_), goal(R',(X,Y)), goalReached(R',T), R!=R'.

#program check(t).

goalReached(R,t) :- rPosition(R,(X,Y),t), goal(R,(X,Y)).

:- not goalReached(R,t), query(t), robot(R).

cost(C,t) :- C = #sum{ T : rPosition(R,_,T), not goalReached(R,T)}.

#minimize{ C : cost(C,t)}. 

#show.
%#show rPosition/3.
%#show goalReached/1.
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).