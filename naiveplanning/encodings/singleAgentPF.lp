%singleAgentPF.lp
%Takes an instance.lp and id (for example -c id = 1), and draws the path for the robot with id 1 to the shelf with the id 1

%#include "selective_input.lp"."
%#include "goal.lp".
%#include "action.lp".
%#include "output.lp".

%goal.lp
processed(A,R) :- ordered(O,A), shelved(S,A), isRobot(R), position(S,C,0),
                  position(R,C,horizon).
processed(A)   :- processed(A,R).

:- ordered(O,A), not processed(A).

%action.lp
time(1..horizon).

direction((X,Y)) :- X=-1..1, Y=-1..1, |X+Y|=1.
nextto((X,Y),(DX,DY),(X',Y')) :- direction((DX,DY)), position((X,Y)), position((X',Y')),
                                 (X,Y)=(X'-DX,Y'-DY), (X',Y')=(X+DX,Y+DY).

{ move(R,D,T) : direction(D) } 1 :- isRobot(R), time(T).

% - move/3 ----------------------------------------------------------------------
position(R,C,T) :- move(R,D,T), position(R,C',T-1),     nextto(C',D,C).
                :- move(R,D,T), position(R,C ,T-1), not nextto(C ,D,_).

% - inertia ---------------------------------------------------------------------
position(R,C,T) :- position(R,C,T-1), not move(R,_,T), isRobot(R), time(T).

% - auxiliaries -----------------------------------------------------------------
 :- { position(R,C,T) } != 1, isRobot(R), time(T).    % REDUNDANT but PERFORMANT?

 %selective_input.lp
 % --------------------------------------------------------------------------------
% REPRESENTATION

robot(R)                     :- init(object(robot,R),          _), R == id.
shelf(S)                     :- init(object(shelf,S),          _), S == id.
station(P)                   :- init(object(pickingStation,P), _), P == id.
product(A)                   :- init(object(product,A),        _), P == id.

    isRobot(robot(R)) :- robot(R), R == id.
    isShelf(shelf(S)) :- shelf(S), S == id.
isStation(station(T)) :- station(T).
isProduct(product(A)) :- product(A), A == id.
    isOrder(order(O)) :- order(O), O == id.

  order(      O            ) :- init(object(order,O),          _), O == id.
ordered(order(O),product(A)) :- init(object(order,O),          value(line,(A,_))), A == id, O == id.      % IGNORING QUANTITIES
 target(order(O),station(P)) :- init(object(order,O),          value(pickingStation,P)), P == id, O == id.

shelved(shelf(S),product(A)) :- init(object(product,A),        value(on,(S,_))), A == id, O == id.        % IGNORING QUANTITIES

position(           (X,Y))   :- init(object(node,_),           value(at,(X,Y))).

position(station(P),(X,Y))   :- init(object(pickingStation,P), value(at,(X,Y))), P == id.

position(  robot(R),(X,Y),0) :- init(object(robot,R),          value(at,(X,Y))), R == id.
position(  shelf(S),(X,Y),0) :- init(object(shelf,S),          value(at,(X,Y))), R == id.

highway(            C    )   :- init(object(highway,_),        value(at,C)).

%output.lp
#show.
#show init/2.


occurs(object(robot,R), action(move,D),     T) :    move(robot(R),D,T).
#show occurs(object(robot,R), action(move,D),     T) :    move(robot(R),D,T).
