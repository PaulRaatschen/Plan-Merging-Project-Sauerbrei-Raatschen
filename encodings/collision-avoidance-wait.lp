%wait.lp
%Input: lp file with rPosition as well as occurs
%In case of a vertex collision occuring between two robots, make the one with the shorter path wait
%

%Determines the longest time period a robot needs to get from his starting position to his end position in the given instance
longestTime(T) :-  T = #max{D: rPosition(R,(X,Y),D)}.

%In the case where a vertex collision appears a robot involved in it gets choosen by the length of their time, with the goal to keep the total time all plans need to a minimum
{choice(R) : vertextCollision((X,Y),T,R,R') , rPosition(R',(X'',Y''),TLongest), longestTime(TLongest)} = 1.

%Rewriting the plan of the robot which doesn't has to wait back into occurs
occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPosition(R,(X,Y), T),rPosition(R,(X',Y'), T+1),DX = X'-X,DY=Y'-Y, choice(R'), R!=R'.


%Rewriting the plan of the robot which does have to wait by adding an extra occur where he doesn't move at all

    %Writing occurs until before the collision
    occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPosition(R,(X,Y), T),rPosition(R,(X',Y'), T+1),DX = X'-X,DY=Y'-Y,vertextCollision((X'',Y''),T',R,R'), choice(R), T+1<T'.
    %Adding the wait for the moment where the crash would occure
    occurs(object(robot,R),action(move,(0,0)),T) :- vertextCollision((X',Y'),T,R,R'), choice(R).
    %Resuming the occurs from after the waiting
    occurs(object(robot,R),action(move,(DX,DY)),T+1) :- rPosition(R,(X,Y), T),rPosition(R,(X',Y'), T+1),DX = X'-X,DY=Y'-Y,vertextCollision((X'',Y''),T',R,R'), choice(R), T=T'.
    occurs(object(robot,R),action(move,(DX,DY)),T+2) :- rPosition(R,(X,Y), T),rPosition(R,(X',Y'), T+1),DX = X'-X,DY=Y'-Y,vertextCollision((X'',Y''),T',R,R'), choice(R), T>=T'.

#show init/2.

#show occurs/3.
