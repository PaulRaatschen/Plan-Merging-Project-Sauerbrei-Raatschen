#include <incmode>.

#program base.
#include setup.lp


#program detectConflicts(t).

conflict((vertex,(X,Y)),T,R,R',t) :- rPosition(R,(X,Y),T,t), rPosition(R',(X,Y),T,t), R!=R'.

conflict((edge,(X,Y),(X',Y')),T,R,R',t) :- rPosition(R,(X,Y),T,t), rPosition(R',(X',Y'),T,t), rPosition(R,(X',Y'),T+1,t), rPosition(R',(X,Y),T+1,t).

:- query(t), conflict(_,_,_,t).

numOfConflicts(N,R,t) :- robot(R), robot(R'), conflict(_,_,R,R',t), R!=R'.

minNumOfConflicts(N,t) :- N = #min { N' : numOfConflicts(N',R,t)}, robot(R).

{finishedPlanning(R,t) : numOfConflicts(N,R,t)} = 1 :- minNumOfConflicts(N).

forbiddenPosition(D,T) :- rPosition(R,D,T,t), finishedPlanning(R,t).

forbiddenMove((X,Y),(DX,DY),T) :- 





#program reschedulePlans(t).

{ move(R,D,T,t+1) : direction(D) } 1 :- robot(R), time(T).

rPosition(R,C,T+1,t+1) :- move(R,D,T,t+1), rPosition(R,C',T,t+1), nextto(C',D,C).
:- move(R,D,T,t+1), position(R,C ,T-1,t), not nextto(C ,D,_).

:- move(R,D,T,_), forbiddenMove(D,T).

:- rPosition(R,D,T,_) :- forbiddenPosition(R,D,T).

rPosition(R,C,T+1,t+1) :- rPosition(R,C,T,t+1), not move(R,_,T,t+1), robot(R), time(T).

 :- { rPosition(R,C,T,t+1) } != 1, robot(R), time(T).    

