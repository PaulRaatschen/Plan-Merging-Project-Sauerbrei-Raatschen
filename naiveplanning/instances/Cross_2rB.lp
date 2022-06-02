%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grid size X: 5
%Grid size Y: 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#program base.

%init
init(object(node, 3), value(at, (3, 1))).
init(object(node, 8), value(at, (3, 2))).
init(object(node, 13), value(at, (3, 3))).
init(object(node, 18), value(at, (3, 4))).
init(object(node, 23), value(at, (3, 5))).
init(object(node, 14), value(at, (4, 3))).
init(object(node, 15), value(at, (5, 3))).
init(object(node, 12), value(at, (2, 3))).
init(object(node, 11), value(at, (1, 3))).
init(object(node, 24), value(at, (4, 5))).
init(object(robot,1), value(at,(3,1))).
init(object(robot,1), value(max_energy,0)).
init(object(robot,1), value(energy,0)).
init(object(robot,2), value(at,(1,3))).
init(object(robot,2), value(max_energy,0)).
init(object(robot,2), value(energy,0)).
init(object(shelf,1), value(at,(4,5))).
init(object(product,1),value(on,(1,1))).
init(object(shelf,2), value(at,(5,3))).
init(object(product,2),value(on,(2,1))).
init(object(order,1),value(pickingStation,0)).
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(pickingStation,0)).
init(object(order,2),value(line,(2,1))).
