import argparse
import random
from turtle import left, right, up


class GenerateInstance:
    def __init__(self, args):


        self.width = args.width

        self.height = args.height

        self.numberOfRobots = args.numberOfRobots

        self.mapType = args.mapType
        self.horizontalLines = args.horizontalLines
        self.verticalLines = args.verticalLines
        self.numberOfRooms = args.numberOfRooms
        self.numberOfWalls = args.numberOfWalls
        self.RobotPositions = []


        self.generate()

    def generate(self):
        
        self.Field = [([0] * self.width) for x in range(self.height)]

        while self.placeWalls() == 0:
            self.Field = [([0] * self.width) for x in range(self.height)]


        self.placeRobots()
        
        self.placeShelves()

        #print(self.Field)

        self.saveField()

        

    def placeWalls(self):

        if self.mapType == "Grid":
            for i in range(0,self.horizontalLines):

                for j in range(0,self.width):
                    y = i+((self.height-self.horizontalLines)//(self.horizontalLines))*(i+1)
                    self.Field[y][j] = 1
                    

            for i in range(0,self.verticalLines):

                for j in range(0,self.height):
                    x = i+((self.width-self.verticalLines)//(self.verticalLines))*(i+1)
                    self.Field[j][x] = 1

            
            for i in range(0,self.height-1):
                if self.Field[i][0] == 1:
                    x = (self.width-self.verticalLines)//(self.verticalLines)//2
                    for j in range(0,self.verticalLines):
                        
                        

                        self.Field[i][x] = 0
                        x = x + (self.width-self.verticalLines)//(self.verticalLines) +1

            for i in range(0,self.width-1):
                if self.Field[0][i] == 1:
                    y = ((self.height-self.horizontalLines)//(self.horizontalLines)//2)
                    for j in range(0,self.horizontalLines):
                        

                        self.Field[y][i] = 0
                        y = y + (self.height-self.horizontalLines)//(self.horizontalLines) + 1
        elif self.mapType == "Random":
            for i in range(0,self.numberOfWalls):
                    while(True):
                        x = random.randint(0,self.width-1)
                        y = random.randint(0,self.height-1)
                        if(self.Field[x][y] != 1):
                            self.Field[x][y] = 1
                            break
                        
        elif self.mapType == "Rooms":
            roomPlaceAttempts = 0

            rooms = []
            for i in range(0,self.numberOfRooms):

                

                while True:
                    
                    roomPlaceAttempts += 1

                    if roomPlaceAttempts > self.numberOfRooms * 20:
                        return 0
                    

                    x1 = random.randint(0,self.width-1)
                    y1 = random.randint(0,self.height-1)
                    x2 = random.randint(0,self.width-1)
                    y2 = random.randint(0,self.height-1)
                    left    = min(x1,x2)
                    right   = max(x1,x2)
                    up      = min(y1,y2)
                    down    = max(y1,y2)

                    

                    if(abs(x1 - x2) > 2 and abs(y1 - y2) > 2):
                        roomValid = True
                        for i in range(left-1,right+1):
                            if (self.Field[up][i] == "FutureWall" or self.Field[down][i] == "FutureWall"
                                or (self.Field[max(up-1,0)][i] == "FutureWall" or self.Field[min(down+1,self.height-1)][i] == "FutureWall")
                                or (self.Field[up+1][i] == "FutureWall" or self.Field[down-1][i] == "FutureWall")):
                                
                                roomValid = False
                                break
                        for i in range(up-1,down+1):
                            if(self.Field[i][left] == 1 or self.Field[i][right] == "FutureWall"
                                or (self.Field[i][max(left-1,0)] == "FutureWall" or self.Field[i][min(right+1,self.width-1)] == "FutureWall")
                                or (self.Field[i][left+1] == "FutureWall" or self.Field[i][right-1] == "FutureWall")):
                                
                                roomValid = False
                                break

                        if(roomValid == True):
                            rooms.append([[left,up],[right,down]])
                            for i in range(left,right+1):
                                self.Field[up][i] = "FutureWall"
                                self.Field[down][i] = "FutureWall"
                            for i in range(up,down+1):
                                self.Field[i][left] = "FutureWall"
                                self.Field[i][right] = "FutureWall"
                            break
                        else:
                            continue
            for room in rooms:

                left = room[0][0]
                right = room[1][0]
                up = room[0][1]
                down = room[1][1]

                for i in range(left,right+1):
                    self.Field[up][i] = 1
                    self.Field[down][i] = 1
                for i in range(up,down+1):
                    self.Field[i][left] = 1
                    self.Field[i][right] = 1

            for room in rooms:
                while True:
                    left = room[0][0]
                    right = room[1][0]
                    up = room[0][1]
                    down = room[1][1]
                    randomint = random.randint(0,3)

                    if randomint == 0:
                        if(up == 0):
                            continue
                        doorplacement = random.randint(left,right)
                        if(self.Field[up+1][doorplacement] == 0 and self.Field[up-1][doorplacement] == 0):
                            self.Field[up][doorplacement] = 0
                            break
                    elif randomint == 1:
                        if(down == self.height-1):
                            continue
                        doorplacement = random.randint(left,right)
                        if(self.Field[down+1][doorplacement] == 0 and self.Field[down-1][doorplacement] == 0):
                            self.Field[down][doorplacement] = 0
                            break
                    elif randomint == 2:
                        if(left == 0):
                            continue
                        doorplacement = random.randint(up,down)
                        
                        if(self.Field[doorplacement][left-1] == 0 and self.Field[doorplacement][left+1] == 0):
                            self.Field[doorplacement][left] = 0
                            break
                    elif randomint == 3:
                        if(right == self.width-1):
                            continue
                        doorplacement = random.randint(up,down)
                        if(self.Field[doorplacement][right-1] == 0 and self.Field[doorplacement][right+1] == 0):
                            self.Field[doorplacement][right] = 0
                            break
        return 1
                    
    def placeRobots(self):
        #Place Robots

        for i in range(0,self.numberOfRobots):
            
            while True:

                randomX = random.randint(0,self.width-1)
                randomY = random.randint(0,self.height-1)
                
                if self.Field[randomY][randomX] == 0:
                        self.Field[randomY][randomX] = ["R" + str(i),"empty"]
                        self.RobotPositions.append([randomX,randomY])
                        break
    
    def reachableSpace(self,x,y,reachableSpaces):
            dx = x+1
            if(dx < self.width):
                if(not [dx,y] in reachableSpaces):
                    if(self.Field[y][dx] != 1):
                        reachableSpaces.append([dx,y])
                        self.reachableSpace(dx,y,reachableSpaces)
            dx = x-1
            if(dx > -1):
                if(not [dx,y] in reachableSpaces):
                    if(self.Field[y][dx] != 1):
                        reachableSpaces.append([dx,y])
                        self.reachableSpace(dx,y,reachableSpaces)

            dy = y+1
            if(dy < self.height):
                if(not [x,dy] in reachableSpaces):
                    if(self.Field[dy][x] != 1):
                        reachableSpaces.append([x,dy])
                        self.reachableSpace(x,dy,reachableSpaces)
            dy = y-1
            if(dy > -1):
                if(not [x,dy] in reachableSpaces):
                    if(self.Field[dy][x] != 1):
                        reachableSpaces.append([x,dy])
                        self.reachableSpace(x,dy,reachableSpaces)

    def placeShelves(self):
        #Place Shelves
        agentCount = -1
        for i in self.RobotPositions:
            agentCount += 1
            possibleSpaces = [[i[0],i[1]]]
            self.reachableSpace(i[0],i[1],possibleSpaces)
            while True:
                pick = random.choice(possibleSpaces)
                x = pick[0]
                y = pick[1]


                if self.Field[y][x] == 0:
                        self.Field[y][x] = ["empty","S" + str(agentCount)]
                        break
                elif self.Field[y][x][1] == "empty":
                        self.Field[y][x][1] = ("S" + str(agentCount))
                        break
    
    


    def saveField(self):
        textFile = "#program base.\n\n%init\n"
        for x in range(0,self.width):
            for y in range(0,self.height):

                if self.Field[y][x] == 1: continue
                
                textFile += "init(object(node, 1), value(at, ("+str(x+1)+", "+str(y+1)+"))).\n"
                
                if type(self.Field[y][x]) == list:
                    if self.Field[y][x][0] != "empty":
                        roboid = str(int(self.Field[y][x][0][1:]) + 1)
                        textFile += "init(object(robot, "+roboid+"), value(at, ("+str(x+1)+", "+str(y+1)+"))).\n"
                        textFile += "init(object(robot, "+roboid+"), value(energy, 0)).\n"
                        textFile += "init(object(robot, "+roboid+"), value(max_energy, 0)).\n"
                    if self.Field[y][x][1] != "empty":
                        shelfid = str(int(self.Field[y][x][1][1:]) + 1)
                        textFile += "init(object(shelf, "+shelfid+"), value(at, ("+str(x+1)+", "+str(y+1)+"))).\n"
                        textFile += "init(object(product, "+shelfid+"), value(on, ("+shelfid+",1))).\n"
                        textFile += "init(object(robot, "+shelfid+"), value(max_energy, 0)).\n"
                        textFile += "init(object(order, "+shelfid+"), value(line, ("+shelfid+", 1))).\n"
                        textFile += "init(object(order, "+shelfid+"), value(pickingstation, 0)).\n"


        with open('generatedInstance.lp', 'w') as f:
            f.write(textFile)
        f.close()


    def createInstance(XSize, YSize, nRobots, roomType, complexity):
        args = argparse.Namespace()

        args.width = XSize
        args.height = YSize
        args.numberOfRobots = nRobots
        args.mapType = roomType
        args.verticalLines = complexity
        args.horizontalLines = complexity
        args.numberOfRooms = complexity
        args.numberOfWalls = complexity
        GenerateInstance(args)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("width", type=int)

    parser.add_argument("height", type=int)

    parser.add_argument("numberOfRobots", type=int)

    parser.add_argument("mapType", type=str)

    parser.add_argument("-hLines","--horizontalLines",default = 3 ,type=int)
    parser.add_argument("-vLines","--verticalLines",default = 3 ,type=int)

    parser.add_argument("-nRooms","--numberOfRooms",default = 3 ,type=int)
    parser.add_argument("-nWalls","--numberOfWalls",default = 3 ,type=int)


    args = parser.parse_args()
    GenerateInstance(args)    

        

    