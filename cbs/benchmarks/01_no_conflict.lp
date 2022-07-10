%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      3
% Grid Size Y:                      3
% Number of Nodes:                  9
% Number of Highway Nodes:          0
% Number of Robots:                 1
% Number of Shelves:                1
% Number of Picking Stations:       0
% Number of Products:               1
% Number of Product Units in Total: 1
% Number of Orders:                 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% init

% nodes
init(object(node,1),value(at,(1,1))).
init(object(node,2),value(at,(1,2))).
init(object(node,3),value(at,(1,3))).
init(object(node,4),value(at,(2,1))).
init(object(node,5),value(at,(2,2))).
init(object(node,6),value(at,(2,3))).
init(object(node,7),value(at,(3,1))).
init(object(node,8),value(at,(3,2))).
init(object(node,9),value(at,(3,3))).

% robots
init(object(robot,1),value(at,(1,3))).

% shelves
init(object(shelf,1),value(at,(3,1))).

% products
init(object(product,1),value(on,(1,1))).

% orders
init(object(order,1),value(line,(1,1))).
occurs(object(robot,1),action(move,(1,0)),1).
occurs(object(robot,1),action(move,(0,-1)),2).
occurs(object(robot,1),action(move,(1,0)),3).
occurs(object(robot,1),action(move,(0,-1)),4).
