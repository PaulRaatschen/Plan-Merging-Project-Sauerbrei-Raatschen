%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Grid size X: 8
%Grid size Y: 6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#program base.

%init
init(object(node, 10), value(at, (2, 2))).
init(object(node, 26), value(at, (2, 4))).
init(object(node, 34), value(at, (2, 5))).
init(object(node, 35), value(at, (3, 5))).
init(object(node, 36), value(at, (4, 5))).
init(object(node, 37), value(at, (5, 5))).
init(object(node, 38), value(at, (6, 5))).
init(object(node, 39), value(at, (7, 5))).
init(object(node, 40), value(at, (8, 5))).
init(object(node, 45), value(at, (5, 6))).
init(object(node, 18), value(at, (2, 3))).
init(object(shelf,1), value(at,(2,2))).
init(object(product,1),value(on,(1,1))).
init(object(shelf,2), value(at,(8,5))).
init(object(product,2),value(on,(2,1))).
init(object(shelf,3), value(at,(7,5))).
init(object(product,3),value(on,(3,1))).
init(object(robot,1), value(at,(6,5))).
init(object(robot,1), value(max_energy,0)).
init(object(robot,1), value(energy,0)).
init(object(robot,3), value(at,(2,3))).
init(object(robot,3), value(max_energy,0)).
init(object(robot,3), value(energy,0)).
init(object(robot,2), value(at,(2,5))).
init(object(robot,2), value(max_energy,0)).
init(object(robot,2), value(energy,0)).
init(object(order,1),value(pickingStation,0)).
init(object(order,1),value(line,(1,1))).
init(object(order,2),value(pickingStation,0)).
init(object(order,2),value(line,(2,1))).
init(object(order,3),value(pickingStation,0)).
init(object(order,3),value(line,(3,1))).
