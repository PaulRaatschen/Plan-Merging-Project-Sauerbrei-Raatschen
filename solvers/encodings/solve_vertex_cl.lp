%wait.lp
%Input: lp file with position as well as inits and possible conflicts
%In case of a vertex conflict occuring between two robots, make the one with the shorter path wait
%

% Takes output of find_conflict.lp as input

earliestVertexConflict(TMIN) :- TMIN = #min{T : conflict(vertex,_,_,_,T)}.

{firstvertextConflict((X0,Y0),T0,R,R'): conflict(vertex,R,R',(X0,Y0),T0), earliestVertexConflict(T0)} = 1.


%Determines the longest time period a robot needs to get from his starting position to his end position in the given instance
longestTime(T) :-  T = #max{D: position(R,(X,Y),D),firstvertextConflict(_,_,R,R')}.

choice(R',prio(2)):- firstvertextConflict((X0,Y0),T0,R,R'), position(R,P1,T0-1), position(R',P1,T0+1),R!=R'.
choice(R',prio(2)):- firstvertextConflict((X0,Y0),T0,R',R), position(R,P1,T0-1), position(R',P1,T0+1),R!=R'.

choice(R',prio(1)):- firstvertextConflict((X0,Y0),T0,R,R'), position(R,P1,T0-1), position(R',P1,T0),R!=R'.
choice(R',prio(1)):- firstvertextConflict((X0,Y0),T0,R',R), position(R,P1,T0-1), position(R',P1,T0),R!=R'.

%In the case where a vertex collision appears a robot involved in it gets choosen by the length of their time, with the goal to keep the total time all plans need to a minimum
{choice(R,prio(0)) : firstvertextConflict((X,Y),T,R,R') , position(R',(X'',Y''),TLongest), longestTime(TLongest)} = 1.

highestPrio(P) :-  P = #max{D: choice(R,prio(D))}.

{finalChoice(R): choice(R,prio(P)), highestPrio(P)} = 1.

#show.
#show position(R,(X,Y),T) : position(R,(X,Y), T), finalChoice(R'), R!=R'.
#show position(R,(X,Y),T) : position(R,(X,Y), T),firstvertextConflict((X'',Y''),T',_,_), finalChoice(R), T<T'.
#show position(R,(X,Y),T) : position(R,(X,Y), T-1),firstvertextConflict((X'',Y''),T,_,_), finalChoice(R).
#show position(R,(X,Y),T+1) : position(R,(X,Y), T),firstvertextConflict((X'',Y''),T',_,_), finalChoice(R), T>=T'.



