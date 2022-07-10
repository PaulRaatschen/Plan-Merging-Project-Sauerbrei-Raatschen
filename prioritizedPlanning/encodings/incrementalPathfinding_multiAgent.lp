%#include <incmode>.
%#include "setup.lp".

#program step(t).

{move(R,D,t) : direction(D)} 1 :- robot(R).

rPosition(R,(X+DX,Y+DY),t) :- rPosition(R,(X,Y),t-1), move(R,(DX,DY),t), robot(R).

rPosition(R,C,t) :- rPosition(R,C,t-1), not move(R,_,t), robot(R).

:- rPosition(R,(X,Y),_), not node(X,Y), robot(R).

% Vertex Conflict
conflict(R,R',t) :- rPosition(R,(X,Y),t), rPosition(R',(X,Y),t), R!=R.

% Edge Conflict
conflict(R,R',t) :- rPosition(R,(X,Y),t-1), move(R,(DX,DY),t), rPosition(R',(X+DX,Y+DY),t-1), rPosition(R',(X,Y),t), R!=R.

% Avoid robots in their goal and passing over r when it is done
:- rPosition(r,(X,Y),t), goal(R,(X,Y)), rPosition(R,(X,Y),T),t >=T, r!=R.
:- rPosition(r,(X,Y),t), goal(r,(X,Y)), rPosition(R,(X,Y),T),t <=T, r!=R.

#program check(t).

goalReached(R,t) :- rPosition(R,(X,Y),t), goal(R,(X,Y)).

:- not goalReached(R,t), query(t), robot(R).

cost(C,t) :- C = #sum{ T : rPosition(R,_,T), not goalReached(R,T)}.

#minimize{ C : cost(C,t)}. 

#show.
%#show rPosition/3.
%#show goalReached/1.
#show occurs(object(robot,R), action(move,D),T) : move(R,D,T).