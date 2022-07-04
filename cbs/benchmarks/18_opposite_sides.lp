%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      7
% Grid Size Y:                      3
% Number of Nodes:                  15
% Number of Highway Nodes:          0
% Number of Robots:                 4
% Number of Shelves:                4
% Number of Picking Stations:       0
% Number of Products:               4
% Number of Product Units in Total: 4
% Number of Orders:                 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% init

% nodes
init(object(node,1),value(at,(1,1))).
init(object(node,2),value(at,(3,1))).
init(object(node,3),value(at,(5,1))).
init(object(node,4),value(at,(7,1))).
init(object(node,5),value(at,(1,2))).
init(object(node,6),value(at,(2,2))).
init(object(node,7),value(at,(3,2))).
init(object(node,8),value(at,(4,2))).
init(object(node,9),value(at,(5,2))).
init(object(node,10),value(at,(6,2))).
init(object(node,11),value(at,(7,2))).
init(object(node,12),value(at,(1,3))).
init(object(node,13),value(at,(3,3))).
init(object(node,14),value(at,(5,3))).
init(object(node,15),value(at,(7,3))).

% robots
init(object(robot,1),value(at,(1,1))).
init(object(robot,3),value(at,(7,1))).
init(object(robot,2),value(at,(1,3))).
init(object(robot,4),value(at,(7,3))).

% shelves
init(object(shelf,3),value(at,(3,1))).
init(object(shelf,1),value(at,(5,1))).
init(object(shelf,4),value(at,(3,3))).
init(object(shelf,2),value(at,(5,3))).

% products
init(object(product,1),value(on,(1,1))).
init(object(product,2),value(on,(2,1))).
init(object(product,3),value(on,(3,1))).
init(object(product,4),value(on,(4,1))).

% orders
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(line,(2,1))).
init(object(order,3),value(line,(3,1))).
init(object(order,4),value(line,(4,1))).

occurs(object(robot,1),action(move,(0,1)),1).
occurs(object(robot,1),action(move,(1,0)),2).
occurs(object(robot,1),action(move,(1,0)),3).
occurs(object(robot,1),action(move,(1,0)),4).
occurs(object(robot,1),action(move,(1,0)),5).
occurs(object(robot,1),action(move,(0,-1)),6).
occurs(object(robot,2),action(move,(0,-1)),1).
occurs(object(robot,2),action(move,(1,0)),2).
occurs(object(robot,2),action(move,(1,0)),3).
occurs(object(robot,2),action(move,(1,0)),4).
occurs(object(robot,2),action(move,(1,0)),5).
occurs(object(robot,2),action(move,(0,1)),6).
occurs(object(robot,3),action(move,(0,1)),1).
occurs(object(robot,3),action(move,(-1,0)),2).
occurs(object(robot,3),action(move,(-1,0)),3).
occurs(object(robot,3),action(move,(-1,0)),4).
occurs(object(robot,3),action(move,(-1,0)),5).
occurs(object(robot,3),action(move,(0,-1)),6).
occurs(object(robot,4),action(move,(0,-1)),1).
occurs(object(robot,4),action(move,(-1,0)),2).
occurs(object(robot,4),action(move,(-1,0)),3).
occurs(object(robot,4),action(move,(-1,0)),4).
occurs(object(robot,4),action(move,(-1,0)),5).
occurs(object(robot,4),action(move,(0,1)),6).
