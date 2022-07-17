%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grid size X: 3
%Grid size Y: 5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#program base.

%init
init(object(node, 2), value(at, (2, 1))).
init(object(node, 5), value(at, (2, 2))).
init(object(node, 14), value(at, (2, 5))).
init(object(node, 3), value(at, (3, 1))).
init(object(node, 6), value(at, (3, 2))).
init(object(node, 9), value(at, (3, 3))).
init(object(node, 12), value(at, (3, 4))).
init(object(node, 15), value(at, (3, 5))).
init(object(node, 4), value(at, (1, 2))).
init(object(node, 7), value(at, (1, 3))).
init(object(node, 10), value(at, (1, 4))).
init(object(node, 13), value(at, (1, 5))).
init(object(node, 1), value(at, (1, 1))).
init(object(robot,1), value(at,(2,1))).
init(object(robot,1), value(max_energy,0)).
init(object(robot,1), value(energy,0)).
init(object(shelf,1), value(at,(2,5))).
init(object(product,1),value(on,(1,1))).
init(object(order,1),value(pickingStation,1)).
init(object(order,1),value(line,(1,1))).