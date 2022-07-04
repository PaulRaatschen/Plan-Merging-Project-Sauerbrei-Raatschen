%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Grid Size X:                      9
% Grid Size Y:                      8
% Number of Nodes:                  44
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
init(object(node,3),value(at,(3,1))).
init(object(node,4),value(at,(7,1))).
init(object(node,5),value(at,(8,1))).
init(object(node,6),value(at,(9,1))).
init(object(node,7),value(at,(2,2))).
init(object(node,8),value(at,(8,2))).
init(object(node,9),value(at,(2,3))).
init(object(node,10),value(at,(3,3))).
init(object(node,11),value(at,(4,3))).
init(object(node,12),value(at,(5,3))).
init(object(node,13),value(at,(6,3))).
init(object(node,14),value(at,(7,3))).
init(object(node,15),value(at,(8,3))).
init(object(node,16),value(at,(2,4))).
init(object(node,17),value(at,(5,4))).
init(object(node,18),value(at,(8,4))).
init(object(node,19),value(at,(1,5))).
init(object(node,20),value(at,(2,5))).
init(object(node,21),value(at,(3,5))).
init(object(node,22),value(at,(5,5))).
init(object(node,23),value(at,(7,5))).
init(object(node,24),value(at,(8,5))).
init(object(node,25),value(at,(9,5))).
init(object(node,26),value(at,(5,6))).
init(object(node,27),value(at,(1,7))).
init(object(node,28),value(at,(2,7))).
init(object(node,29),value(at,(3,7))).
init(object(node,30),value(at,(4,7))).
init(object(node,31),value(at,(5,7))).
init(object(node,32),value(at,(6,7))).
init(object(node,33),value(at,(7,7))).
init(object(node,34),value(at,(8,7))).
init(object(node,35),value(at,(9,7))).
init(object(node,36),value(at,(1,8))).
init(object(node,37),value(at,(2,8))).
init(object(node,38),value(at,(3,8))).
init(object(node,39),value(at,(4,8))).
init(object(node,40),value(at,(5,8))).
init(object(node,41),value(at,(6,8))).
init(object(node,42),value(at,(7,8))).
init(object(node,43),value(at,(8,8))).
init(object(node,44),value(at,(9,8))).

% robots
init(object(robot,1),value(at,(1,1))).
init(object(robot,2),value(at,(3,1))).
init(object(robot,5),value(at,(7,1))).
init(object(robot,6),value(at,(9,1))).
init(object(robot,3),value(at,(1,5))).
init(object(robot,4),value(at,(3,5))).
init(object(robot,7),value(at,(7,5))).
init(object(robot,8),value(at,(9,5))).

% shelves
init(object(shelf,1),value(at,(1,8))).
init(object(shelf,2),value(at,(2,8))).
init(object(shelf,3),value(at,(3,8))).
init(object(shelf,4),value(at,(4,8))).
init(object(shelf,5),value(at,(6,8))).
init(object(shelf,6),value(at,(7,8))).
init(object(shelf,7),value(at,(8,8))).
init(object(shelf,8),value(at,(9,8))).

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
occurs(object(robot,1),action(move,(0,1)),7).
occurs(object(robot,1),action(move,(0,1)),8).
occurs(object(robot,1),action(move,(0,1)),9).
occurs(object(robot,1),action(move,(0,1)),10).
occurs(object(robot,1),action(move,(-1,0)),11).
occurs(object(robot,1),action(move,(-1,0)),12).
occurs(object(robot,1),action(move,(-1,0)),13).
occurs(object(robot,1),action(move,(-1,0)),14).
occurs(object(robot,1),action(move,(0,1)),15).
occurs(object(robot,2),action(move,(-1,0)),1).
occurs(object(robot,2),action(move,(0,1)),2).
occurs(object(robot,2),action(move,(0,1)),3).
occurs(object(robot,2),action(move,(1,0)),4).
occurs(object(robot,2),action(move,(1,0)),5).
occurs(object(robot,2),action(move,(1,0)),6).
occurs(object(robot,2),action(move,(0,1)),7).
occurs(object(robot,2),action(move,(0,1)),8).
occurs(object(robot,2),action(move,(0,1)),9).
occurs(object(robot,2),action(move,(0,1)),10).
occurs(object(robot,2),action(move,(-1,0)),11).
occurs(object(robot,2),action(move,(-1,0)),12).
occurs(object(robot,2),action(move,(-1,0)),13).
occurs(object(robot,2),action(move,(0,1)),14).
occurs(object(robot,3),action(move,(1,0)),1).
occurs(object(robot,3),action(move,(0,-1)),2).
occurs(object(robot,3),action(move,(0,-1)),3).
occurs(object(robot,3),action(move,(1,0)),4).
occurs(object(robot,3),action(move,(1,0)),5).
occurs(object(robot,3),action(move,(1,0)),6).
occurs(object(robot,3),action(move,(0,1)),7).
occurs(object(robot,3),action(move,(0,1)),8).
occurs(object(robot,3),action(move,(0,1)),9).
occurs(object(robot,3),action(move,(0,1)),10).
occurs(object(robot,3),action(move,(-1,0)),11).
occurs(object(robot,3),action(move,(-1,0)),12).
occurs(object(robot,3),action(move,(0,1)),13).
occurs(object(robot,4),action(move,(-1,0)),1).
occurs(object(robot,4),action(move,(0,-1)),2).
occurs(object(robot,4),action(move,(0,-1)),3).
occurs(object(robot,4),action(move,(1,0)),4).
occurs(object(robot,4),action(move,(1,0)),5).
occurs(object(robot,4),action(move,(1,0)),6).
occurs(object(robot,4),action(move,(0,1)),7).
occurs(object(robot,4),action(move,(0,1)),8).
occurs(object(robot,4),action(move,(0,1)),9).
occurs(object(robot,4),action(move,(0,1)),10).
occurs(object(robot,4),action(move,(0,1)),11).
occurs(object(robot,4),action(move,(-1,0)),12).
occurs(object(robot,5),action(move,(1,0)),1).
occurs(object(robot,5),action(move,(0,1)),2).
occurs(object(robot,5),action(move,(0,1)),3).
occurs(object(robot,5),action(move,(-1,0)),4).
occurs(object(robot,5),action(move,(-1,0)),5).
occurs(object(robot,5),action(move,(-1,0)),6).
occurs(object(robot,5),action(move,(0,1)),7).
occurs(object(robot,5),action(move,(0,1)),8).
occurs(object(robot,5),action(move,(0,1)),9).
occurs(object(robot,5),action(move,(0,1)),10).
occurs(object(robot,5),action(move,(0,1)),11).
occurs(object(robot,5),action(move,(1,0)),12).
occurs(object(robot,6),action(move,(-1,0)),1).
occurs(object(robot,6),action(move,(0,1)),2).
occurs(object(robot,6),action(move,(0,1)),3).
occurs(object(robot,6),action(move,(-1,0)),4).
occurs(object(robot,6),action(move,(-1,0)),5).
occurs(object(robot,6),action(move,(-1,0)),6).
occurs(object(robot,6),action(move,(0,1)),7).
occurs(object(robot,6),action(move,(0,1)),8).
occurs(object(robot,6),action(move,(0,1)),9).
occurs(object(robot,6),action(move,(0,1)),10).
occurs(object(robot,6),action(move,(1,0)),11).
occurs(object(robot,6),action(move,(1,0)),12).
occurs(object(robot,6),action(move,(0,1)),13).
occurs(object(robot,7),action(move,(1,0)),1).
occurs(object(robot,7),action(move,(0,-1)),2).
occurs(object(robot,7),action(move,(0,-1)),3).
occurs(object(robot,7),action(move,(-1,0)),4).
occurs(object(robot,7),action(move,(-1,0)),5).
occurs(object(robot,7),action(move,(-1,0)),6).
occurs(object(robot,7),action(move,(0,1)),7).
occurs(object(robot,7),action(move,(0,1)),8).
occurs(object(robot,7),action(move,(0,1)),9).
occurs(object(robot,7),action(move,(0,1)),10).
occurs(object(robot,7),action(move,(1,0)),11).
occurs(object(robot,7),action(move,(1,0)),12).
occurs(object(robot,7),action(move,(1,0)),13).
occurs(object(robot,7),action(move,(0,1)),14).
occurs(object(robot,8),action(move,(-1,0)),1).
occurs(object(robot,8),action(move,(0,-1)),2).
occurs(object(robot,8),action(move,(0,-1)),3).
occurs(object(robot,8),action(move,(-1,0)),4).
occurs(object(robot,8),action(move,(-1,0)),5).
occurs(object(robot,8),action(move,(-1,0)),6).
occurs(object(robot,8),action(move,(0,1)),7).
occurs(object(robot,8),action(move,(0,1)),8).
occurs(object(robot,8),action(move,(0,1)),9).
occurs(object(robot,8),action(move,(0,1)),10).
occurs(object(robot,8),action(move,(1,0)),11).
occurs(object(robot,8),action(move,(1,0)),12).
occurs(object(robot,8),action(move,(1,0)),13).
occurs(object(robot,8),action(move,(1,0)),14).
occurs(object(robot,8),action(move,(0,1)),15).
