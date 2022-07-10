%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      9
% Grid Size Y:                      5
% Number of Nodes:                  33
% Number of Highway Nodes:          0
% Number of Robots:                 8
% Number of Shelves:                8
% Number of Picking Stations:       0
% Number of Products:               8
% Number of Product Units in Total: 8
% Number of Orders:                 8
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#program base.

% init

% nodes
init(object(node,1),value(at,(1,1))).
init(object(node,2),value(at,(2,1))).
init(object(node,3),value(at,(4,1))).
init(object(node,4),value(at,(6,1))).
init(object(node,5),value(at,(8,1))).
init(object(node,6),value(at,(9,1))).
init(object(node,7),value(at,(1,2))).
init(object(node,8),value(at,(2,2))).
init(object(node,9),value(at,(4,2))).
init(object(node,10),value(at,(6,2))).
init(object(node,11),value(at,(8,2))).
init(object(node,12),value(at,(9,2))).
init(object(node,13),value(at,(1,3))).
init(object(node,14),value(at,(2,3))).
init(object(node,15),value(at,(3,3))).
init(object(node,16),value(at,(4,3))).
init(object(node,17),value(at,(5,3))).
init(object(node,18),value(at,(6,3))).
init(object(node,19),value(at,(7,3))).
init(object(node,20),value(at,(8,3))).
init(object(node,21),value(at,(9,3))).
init(object(node,22),value(at,(1,4))).
init(object(node,23),value(at,(2,4))).
init(object(node,24),value(at,(4,4))).
init(object(node,25),value(at,(6,4))).
init(object(node,26),value(at,(8,4))).
init(object(node,27),value(at,(9,4))).
init(object(node,28),value(at,(1,5))).
init(object(node,29),value(at,(2,5))).
init(object(node,30),value(at,(4,5))).
init(object(node,31),value(at,(6,5))).
init(object(node,32),value(at,(8,5))).
init(object(node,33),value(at,(9,5))).

% robots
init(object(robot,1),value(at,(1,1))).
init(object(robot,5),value(at,(9,1))).
init(object(robot,2),value(at,(1,2))).
init(object(robot,6),value(at,(9,2))).
init(object(robot,3),value(at,(1,4))).
init(object(robot,7),value(at,(9,4))).
init(object(robot,4),value(at,(1,5))).
init(object(robot,8),value(at,(9,5))).

% shelves
init(object(shelf,5),value(at,(4,1))).
init(object(shelf,1),value(at,(6,1))).
init(object(shelf,6),value(at,(4,2))).
init(object(shelf,2),value(at,(6,2))).
init(object(shelf,7),value(at,(4,4))).
init(object(shelf,3),value(at,(6,4))).
init(object(shelf,8),value(at,(4,5))).
init(object(shelf,4),value(at,(6,5))).

% products
init(object(product,1),value(on,(1,1))).
init(object(product,2),value(on,(2,1))).
init(object(product,3),value(on,(3,1))).
init(object(product,4),value(on,(4,1))).
init(object(product,5),value(on,(5,1))).
init(object(product,6),value(on,(6,1))).
init(object(product,7),value(on,(7,1))).
init(object(product,8),value(on,(8,1))).

% orders
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(line,(2,1))).
init(object(order,3),value(line,(3,1))).
init(object(order,4),value(line,(4,1))).
init(object(order,5),value(line,(5,1))).
init(object(order,6),value(line,(6,1))).
init(object(order,7),value(line,(7,1))).
init(object(order,8),value(line,(8,1))).

occurs(object(robot,1),action(move,(1,0)),1).
occurs(object(robot,1),action(move,(0,1)),2).
occurs(object(robot,1),action(move,(0,1)),3).
occurs(object(robot,1),action(move,(1,0)),4).
occurs(object(robot,1),action(move,(1,0)),5).
occurs(object(robot,1),action(move,(1,0)),6).
occurs(object(robot,1),action(move,(1,0)),7).
occurs(object(robot,1),action(move,(0,-1)),8).
occurs(object(robot,1),action(move,(0,-1)),9).
occurs(object(robot,2),action(move,(1,0)),1).
occurs(object(robot,2),action(move,(0,1)),2).
occurs(object(robot,2),action(move,(1,0)),3).
occurs(object(robot,2),action(move,(1,0)),4).
occurs(object(robot,2),action(move,(1,0)),5).
occurs(object(robot,2),action(move,(1,0)),6).
occurs(object(robot,2),action(move,(0,-1)),7).
occurs(object(robot,3),action(move,(1,0)),1).
occurs(object(robot,3),action(move,(0,-1)),2).
occurs(object(robot,3),action(move,(1,0)),3).
occurs(object(robot,3),action(move,(1,0)),4).
occurs(object(robot,3),action(move,(1,0)),5).
occurs(object(robot,3),action(move,(1,0)),6).
occurs(object(robot,3),action(move,(0,1)),7).
occurs(object(robot,4),action(move,(1,0)),1).
occurs(object(robot,4),action(move,(0,-1)),2).
occurs(object(robot,4),action(move,(0,-1)),3).
occurs(object(robot,4),action(move,(1,0)),4).
occurs(object(robot,4),action(move,(1,0)),5).
occurs(object(robot,4),action(move,(1,0)),6).
occurs(object(robot,4),action(move,(1,0)),7).
occurs(object(robot,4),action(move,(0,1)),8).
occurs(object(robot,4),action(move,(0,1)),9).
occurs(object(robot,5),action(move,(-1,0)),1).
occurs(object(robot,5),action(move,(0,1)),2).
occurs(object(robot,5),action(move,(0,1)),3).
occurs(object(robot,5),action(move,(-1,0)),4).
occurs(object(robot,5),action(move,(-1,0)),5).
occurs(object(robot,5),action(move,(-1,0)),6).
occurs(object(robot,5),action(move,(-1,0)),7).
occurs(object(robot,5),action(move,(0,-1)),8).
occurs(object(robot,5),action(move,(0,-1)),9).
occurs(object(robot,6),action(move,(-1,0)),1).
occurs(object(robot,6),action(move,(0,1)),2).
occurs(object(robot,6),action(move,(-1,0)),3).
occurs(object(robot,6),action(move,(-1,0)),4).
occurs(object(robot,6),action(move,(-1,0)),5).
occurs(object(robot,6),action(move,(-1,0)),6).
occurs(object(robot,6),action(move,(0,-1)),7).
occurs(object(robot,7),action(move,(-1,0)),1).
occurs(object(robot,7),action(move,(0,-1)),2).
occurs(object(robot,7),action(move,(-1,0)),3).
occurs(object(robot,7),action(move,(-1,0)),4).
occurs(object(robot,7),action(move,(-1,0)),5).
occurs(object(robot,7),action(move,(-1,0)),6).
occurs(object(robot,7),action(move,(0,1)),7).
occurs(object(robot,8),action(move,(-1,0)),1).
occurs(object(robot,8),action(move,(0,-1)),2).
occurs(object(robot,8),action(move,(0,-1)),3).
occurs(object(robot,8),action(move,(-1,0)),4).
occurs(object(robot,8),action(move,(-1,0)),5).
occurs(object(robot,8),action(move,(-1,0)),6).
occurs(object(robot,8),action(move,(-1,0)),7).
occurs(object(robot,8),action(move,(0,1)),8).
occurs(object(robot,8),action(move,(0,1)),9).
