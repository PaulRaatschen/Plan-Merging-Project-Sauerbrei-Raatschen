% Handles the solving a vertex conflicts

% Setup

    earliestVertexConflict(TMIN) :- TMIN = #min{T : conflict(vertex,_,_,_,T)}.
    {firstvertextConflict((X0,Y0),T0,R,R'): conflict(vertex,R,R',(X0,Y0),T0), earliestVertexConflict(T0)} = 1.


% Determines the longest time period a robot needs to get from his starting position to his end position in the given instance
    longestTime(T) :-  T = #max{D: position(R,(X,Y),D),firstvertextConflict(_,_,R,R')}.

% Determine highest priority of which agent has to wait.

    % Priority 2: One Agent wants to continue his path onto the path of the other agent
    choice(R',prio(2)):- firstvertextConflict((X0,Y0),T0,R,R'), position(R,P1,T0-1), position(R',P1,T0+1),R!=R'.
    choice(R',prio(2)):- firstvertextConflict((X0,Y0),T0,R',R), position(R,P1,T0-1), position(R',P1,T0+1),R!=R'.

    % Priority 0: One Agent has a longer path than the other one
    {choice(R,prio(0)) : firstvertextConflict((X,Y),T,R,R') , position(R',(X'',Y''),TLongest), longestTime(TLongest)} = 1.

    % Determine highest priority in the conflict
    highestPrio(P) :-  P = #max{D: choice(R,prio(D))}.

    % Choose agent which has to wait based on the priority
    {finalChoice(R): choice(R,prio(P)), highestPrio(P)} = 1.

% Show the new paths of the agents
    #show.

    % Non waiting agents
        #show position(R,(X,Y),T) : position(R,(X,Y), T), finalChoice(R'), R!=R'.
    % Waiting agent
        #show position(R,(X,Y),T) : position(R,(X,Y), T),firstvertextConflict((X'',Y''),T',_,_), finalChoice(R), T<T'.
        #show position(R,(X,Y),T) : position(R,(X,Y), T-1),firstvertextConflict((X'',Y''),T,_,_), finalChoice(R).
        #show position(R,(X,Y),T+1) : position(R,(X,Y), T),firstvertextConflict((X'',Y''),T',_,_), finalChoice(R), T>=T'.