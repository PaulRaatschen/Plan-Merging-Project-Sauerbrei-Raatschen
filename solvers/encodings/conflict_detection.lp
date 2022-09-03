
% Create new positions to check if agents go over each other when they are at their goal node
    lastTime(LT):- LT = #max{T : position(_,_,T)}.
    lastTimeAgent(R,P,LT):- LT = #max{T : position(R,_,T)}, position(R,P,LT).
    
    pastGoal(R,P,TT+1):- lastTimeAgent(R,P,TT), TT<= LT, lastTime(LT).
    pastGoal(R,P,T+1):- lastTimeAgent(R,P,TT), pastGoal(R,P,T),TT<=T ,T< LT, lastTime(LT).


% Vertex Conflict
conflict(vertex,R,R',(X,Y),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', position(R,(X0,Y0),T-1),position(R',(X0',Y0'),T-1), not position(R,(X0',Y0'),T+1),not position(R',(X0,Y0),T+1).

% Edge Conflict
    %Both agents switch position
    conflict(edge,R,R',((X,Y),(X',Y')),T) :-  position(R,(X,Y),T), position(R',(X',Y'),T), position(R,(X',Y'),T+1), position(R',(X,Y),T+1), R != R',|X-X'|+|Y-Y'| > 0.
    %Both agent go onto the same spot
    conflict(edge,R,R',((X0,Y0),(X1,Y1)),T) :- position(R,(X,Y),T), position(R',(X,Y),T), R!=R', position(R,(X1,Y1),T+1), position(R',(X1,Y1),T-1), position(R,(X0,Y0),T-1), position(R',(X0,Y0),T+1), R != R',|X0-X1|+|Y0-Y1| > 0.
    %One agent waits while the other goes into him
    conflict(edge,R,R',(P,P'),T+1):- position(R,P,T), position(R,P,T+1),position(R',P',T), position(R',P,T+1), R != R', P != P'.

% Show the found conflicts expand the position of every agent past the moment they reached their goal
    #show conflict/5.
    #show position(R,P,T) : pastGoal(R,P,T), not conflict(edge,_,_,_,_).

