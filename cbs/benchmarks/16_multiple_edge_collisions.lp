%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      4
% Grid Size Y:                      5
% Number of Nodes:                  13
% Number of Highway Nodes:          0
% Number of Robots:                 3
% Number of Shelves:                3
% Number of Picking Stations:       0
% Number of Products:               3
% Number of Product Units in Total: 3
% Number of Orders:                 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% init

% nodes
init(object(node,1),value(at,(1,1))).
init(object(node,2),value(at,(2,1))).
init(object(node,3),value(at,(3,1))).
init(object(node,4),value(at,(4,1))).
init(object(node,5),value(at,(2,2))).
init(object(node,6),value(at,(3,2))).
init(object(node,7),value(at,(2,3))).
init(object(node,8),value(at,(3,3))).
init(object(node,9),value(at,(1,4))).
init(object(node,10),value(at,(2,4))).
init(object(node,11),value(at,(3,4))).
init(object(node,12),value(at,(4,4))).
init(object(node,13),value(at,(2,5))).

% robots
init(object(robot,3),value(at,(2,1))).
init(object(robot,1),value(at,(1,4))).
init(object(robot,2),value(at,(4,4))).

% shelves
init(object(shelf,2),value(at,(1,1))).
init(object(shelf,1),value(at,(4,1))).
init(object(shelf,3),value(at,(2,5))).

% products
init(object(product,1),value(on,(1,1))).
init(object(product,2),value(on,(2,1))).
init(object(product,3),value(on,(3,1))).

% orders
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(line,(2,1))).
init(object(order,3),value(line,(3,1))).

occurs(object(robot,1),action(move,(1,0)),1).
occurs(object(robot,1),action(move,(1,0)),2).
occurs(object(robot,1),action(move,(0,-1)),3).
occurs(object(robot,1),action(move,(0,-1)),4).
occurs(object(robot,1),action(move,(0,-1)),5).
occurs(object(robot,1),action(move,(1,0)),6).
occurs(object(robot,2),action(move,(-1,0)),1).
occurs(object(robot,2),action(move,(-1,0)),2).
occurs(object(robot,2),action(move,(0,-1)),3).
occurs(object(robot,2),action(move,(0,-1)),4).
occurs(object(robot,2),action(move,(0,-1)),5).
occurs(object(robot,2),action(move,(-1,0)),6).
occurs(object(robot,3),action(move,(0,1)),1).
occurs(object(robot,3),action(move,(0,1)),2).
occurs(object(robot,3),action(move,(0,1)),3).
occurs(object(robot,3),action(move,(0,1)),4).
