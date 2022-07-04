%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      5
% Grid Size Y:                      5
% Number of Nodes:                  9
% Number of Highway Nodes:          9
% Number of Robots:                 2
% Number of Shelves:                2
% Number of Picking Stations:       0
% Number of Products:               2
% Number of Product Units in Total: 2
% Number of Orders:                 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% init

% nodes
init(object(node,1),value(at,(3,1))).
init(object(node,2),value(at,(3,2))).
init(object(node,3),value(at,(3,3))).
init(object(node,4),value(at,(3,4))).
init(object(node,5),value(at,(3,5))).
init(object(node,6),value(at,(1,3))).
init(object(node,7),value(at,(2,3))).
init(object(node,8),value(at,(4,3))).
init(object(node,9),value(at,(5,3))).

% highways
init(object(highway,1),value(at,(3,1))).
init(object(highway,2),value(at,(3,2))).
init(object(highway,3),value(at,(3,3))).
init(object(highway,4),value(at,(3,4))).
init(object(highway,5),value(at,(3,5))).
init(object(highway,6),value(at,(1,3))).
init(object(highway,7),value(at,(2,3))).
init(object(highway,8),value(at,(4,3))).
init(object(highway,9),value(at,(5,3))).


% robots
init(object(robot,1),value(at,(1,3))).
init(object(robot,2),value(at,(3,5))).


% shelves
init(object(shelf,1),value(at,(5,3))).
init(object(shelf,2),value(at,(3,1))).

% products
init(object(product,1),value(on,(1,1))).
init(object(product,2),value(on,(2,1))).

% orders
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(line,(2,1))).

occurs(object(robot,1),action(move,(1,0)),1).
occurs(object(robot,1),action(move,(1,0)),2).
occurs(object(robot,1),action(move,(1,0)),3).
occurs(object(robot,1),action(move,(1,0)),4).
occurs(object(robot,2),action(move,(0,-1)),1).
occurs(object(robot,2),action(move,(0,-1)),2).
occurs(object(robot,2),action(move,(0,-1)),3).
occurs(object(robot,2),action(move,(0,-1)),4).
