%wait.lp
%Input: lp file with rPositionX as well as inits and possible conflicts
%In case of a vertex collision occuring between two robots, make the one with the shorter path wait
%

% Takes output of find_conflict.lp as input

edgeCollision(1,1,1,1,1,1).

earliestVertexCollision(TMIN) :- TMIN = #min{T : vertextCollision(_,T,_,_)}.

{firstvertextCollision((X0,Y0),T0,R,R'): vertextCollision((X0,Y0),T0,R,R'), earliestVertexCollision(T0)} = 1.


%Determines the longest time period a robot needs to get from his starting position to his end position in the given instance
longestTime(T) :-  T = #max{D: rPositionX(R,(X,Y),D),firstvertextCollision(_,_,R,R')}.

choice(R',prio(2)):- firstvertextCollision((X0,Y0),T0,R,R'), rPositionX(R,P1,T0-1), rPositionX(R',P1,T0+1),R!=R'.
choice(R',prio(2)):- firstvertextCollision((X0,Y0),T0,R',R), rPositionX(R,P1,T0-1), rPositionX(R',P1,T0+1),R!=R'.



choice(R',prio(1)):- firstvertextCollision((X0,Y0),T0,R,R'), rPositionX(R,P1,T0-1), rPositionX(R',P1,T0),R!=R'.
choice(R',prio(1)):- firstvertextCollision((X0,Y0),T0,R',R), rPositionX(R,P1,T0-1), rPositionX(R',P1,T0),R!=R'.

%In the case where a vertex collision appears a robot involved in it gets choosen by the length of their time, with the goal to keep the total time all plans need to a minimum
{choice(R,prio(0)) : firstvertextCollision((X,Y),T,R,R') , rPositionX(R',(X'',Y''),TLongest), longestTime(TLongest)} = 1.

highestPrio(P) :-  P = #max{D: choice(R,prio(D))}.

{finalChoice(R): choice(R,prio(P)), highestPrio(P)} = 1.

%Rewriting the plan of the robot which doesn't has to wait back into occurs
rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y), T), finalChoice(R'), R!=R'.


%Rewriting the plan of the robot which does have to wait by adding an extra occur where he doesn't move at all

    %Writing occurs until before the collision
    rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y), T),firstvertextCollision((X'',Y''),T',_,_), finalChoice(R), T<T'.
    %Adding the wait for the moment where the crash would occure
    rPosition(R,(X,Y),T) :- rPositionX(R,(X,Y), T-1),firstvertextCollision((X'',Y''),T,_,_), finalChoice(R).
    %Resuming the occurs from after the waiting
    rPosition(R,(X,Y),T+1) :- rPositionX(R,(X,Y), T),firstvertextCollision((X'',Y''),T',_,_), finalChoice(R), T>=T'.


#show init/2.
#show rPosition/3.
#show firstvertextCollision/4.
#show choice/2.
#show finalChoice/1.



