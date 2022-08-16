%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grid size X: 6
%Grid size Y: 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#program base.

%init
init(object(node, 7), value(at, (1, 2))).
init(object(node, 8), value(at, (2, 2))).
init(object(node, 9), value(at, (3, 2))).
init(object(node, 10), value(at, (4, 2))).
init(object(node, 11), value(at, (5, 2))).
init(object(node, 12), value(at, (6, 2))).
init(object(node, 17), value(at, (5, 3))).
init(object(node, 30), value(at, (6, 5))).
init(object(node, 29), value(at, (5, 5))).
init(object(node, 28), value(at, (4, 5))).
init(object(node, 27), value(at, (3, 5))).
init(object(node, 26), value(at, (2, 5))).
init(object(node, 25), value(at, (1, 5))).
init(object(node, 35), value(at, (5, 6))).
init(object(robot,1), value(at,(5,2))).
init(object(robot,1), value(max_energy,0)).
init(object(robot,1), value(energy,0)).
init(object(robot,2), value(at,(2,2))).
init(object(robot,2), value(max_energy,0)).
init(object(robot,2), value(energy,0)).
init(object(robot,3), value(at,(2,5))).
init(object(robot,3), value(max_energy,0)).
init(object(robot,3), value(energy,0)).
init(object(shelf,1), value(at,(1,2))).
init(object(product,1),value(on,(1,1))).
init(object(shelf,2), value(at,(6,2))).
init(object(product,2),value(on,(2,1))).
init(object(shelf,3), value(at,(6,5))).
init(object(product,3),value(on,(3,1))).
init(object(shelf,4), value(at,(1,5))).
init(object(product,4),value(on,(4,1))).
init(object(order,1),value(pickingStation,0)).
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(pickingStation,0)).
init(object(order,2),value(line,(2,1))).
init(object(order,3),value(pickingStation,0)).
init(object(order,3),value(line,(3,1))).
init(object(order,4),value(pickingStation,0)).
init(object(order,4),value(line,(4,1))).
