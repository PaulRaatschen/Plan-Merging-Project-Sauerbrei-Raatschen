%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grid size X: 11
%Grid size Y: 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#program base.

%init
init(object(node, 12), value(at, (1, 2))).
init(object(node, 13), value(at, (2, 2))).
init(object(node, 14), value(at, (3, 2))).
init(object(node, 15), value(at, (4, 2))).
init(object(node, 16), value(at, (5, 2))).
init(object(node, 17), value(at, (6, 2))).
init(object(node, 18), value(at, (7, 2))).
init(object(node, 19), value(at, (8, 2))).
init(object(node, 20), value(at, (9, 2))).
init(object(node, 21), value(at, (10, 2))).
init(object(node, 22), value(at, (11, 2))).
init(object(node, 8), value(at, (8, 1))).
init(object(robot,1), value(at,(10,2))).
init(object(robot,1), value(max_energy,0)).
init(object(robot,1), value(energy,0)).
init(object(robot,2), value(at,(2,2))).
init(object(robot,2), value(max_energy,0)).
init(object(robot,2), value(energy,0)).
init(object(shelf,1), value(at,(1,2))).
init(object(product,1),value(on,(1,1))).
init(object(shelf,2), value(at,(11,2))).
init(object(product,2),value(on,(2,1))).
init(object(order,1),value(pickingStation,0)).
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(pickingStation,0)).
init(object(order,2),value(line,(2,1))).
