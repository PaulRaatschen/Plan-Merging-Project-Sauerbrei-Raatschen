#program base.

#program robot(r).

{move(r,D,T) : direction(D)} 1 :- time(T).

rPosition(r,(X+DX,Y+DY),T) :- rPosition(r,(X,Y),T-1), move(r,(DX,DY),T).

rPosition(r,C,T) :- rPosition(r,C,T-1), not move(r,_,T), time(T).

:- rPosition(r,(X,Y),_), not node(X,Y).

goalReached(r) :- rPosition(r,(X,Y),H), goal(r,(X,Y)), horizon(H).

:- not goalReached(r).

% Vertex Conflict
:- rPosition(r,(X,Y),T), rPosition(R,(X,Y),T), R!=r.

% Edge Conflict
:- rPosition(r,(X,Y),T), move(r,(DX,DY),T+1), rPosition(R,(X+DX,Y+DY),T), rPosition(R,(X,Y),T+1), R!=r.

moves(r,N) :- N = #sum{ T : move(r,_,T) }.

#minimize{ N : moves(r,N) }.

#show.
#show rPosition/3.
#show occurs(object(robot,r), action(move,D),T) : move(r,D,T).