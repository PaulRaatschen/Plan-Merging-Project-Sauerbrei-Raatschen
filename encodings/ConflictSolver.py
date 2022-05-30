"""
ConflictSolver.py
Input:  File can become up to 4 clingo files as input: "instance.lp" singleAgentPF.lp conflict_detection.lp "conflictsolving.lp"
        Yet it also works when giving only instance.lp, then the program has to run in the main folder
example in Anaconda:
python ConflictSolver.py Cross_2rB.lp singleAgentPF.lp conflict_detection.lp collision-avoidance.lp
Output: Creates a new directory, named after the instance,  containing a file which contains all paths,
                                                            containing a file with the conflict model
                                                            & containing a file with the new path solution.
Needs to run in an environment which supports clingo applications
"""

import re
import sys #used to give .lp files in the prompt
import ntpath
import clingo
from os import listdir, mkdir #to create a folder which saves the results
from os import system #To ropen viz after executing system('viz')
from collections import Counter #used to remove duplicates


resultOfClingo = ""

edgeIterations = 200

vertexIterations = 100

ListOfConflicts = []

MultipleTimesOccuringConflicts = []

#standard clingo function
class Application:
    def __init__(self, name):
        self.program_name = name

    def main(self, ctl, files):

        

        if len(files) > 0:
            for f in files:
                ctl.load(f)
        else:
            ctl.load("-")
        ctl.ground([("base", [])])
        
        with ctl.solve(yield_=True) as handle:
            #Saves resulst into global string resultOfClingo
            global resultOfClingo
            for m in handle: resultOfClingo = format(m).replace(" ", ".") + "."

#removes duplicate outputs -> Used to unite all paths
def remove_duplicates(input):
 
    # split input string separated by space
    input = input.split(".")
 
    # now create dictionary using counter method
    # which will have strings as key and their
    # frequencies as value
    UniqW = Counter(input)
 
    # joins two adjacent elements in iterable way
    s = ".".join(UniqW.keys())
    return(s)

#function which helps getting the name of the instance
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def solveEdge():
    global edgeIterations
    while(edgeIterations > 0):

        global resultOfClingo
        global resultLocation
        global colissionLocation
        global inputs
        global ListOfConflicts
        global MultipleTimesOccuringConflicts
        #creates and saves all conflicts
        c = clingo.clingo_main(Application(sys.argv[0]), [inputs[2],resultLocation ,"--outf=3"])


        text_file = open(colissionLocation, "w")
        text_file.write(resultOfClingo)
        text_file.close()
        edgeIterations = edgeIterations -1
        if "edge" in resultOfClingo:
                #creates and saves the found solution
            c = clingo.clingo_main(Application(sys.argv[0]), [inputs[3],colissionLocation ,"--outf=3"])
            
            if "edge" in resultOfClingo:

                print("No solution in itearation " + str(i))
                
                for j in resultOfClingo.split("."):
                
                    if "Col" in j:
                    
                        print(j + "\n")
                
                break


                
                # c = clingo.clingo_main(Application(sys.argv[0]), [inputs[4],colissionLocation ,"--outf=3"])
                # if "edge" in resultOfClingo:
                #      print("\nExtra vertex collision didn't work")
                #      continue
                # print("\n Vertex Collision solved")
                # text_file = open(resultLocation, "w")
                # text_file.write(resultOfClingo)
                # text_file.close()

        

            print("\n\nEdge - Iteration " + str(edgeIterations)+ "\n")
            for j in resultOfClingo.split("."):
                
                if "conflict" in j:
                    
                    if j in ListOfConflicts:
                        print("R")
                        MultipleTimesOccuringConflicts.append(j+".")
                    ListOfConflicts.append(j)
                    print(j + "\n")
                if "wa" in j:
                    print(j)

            #resultOfClingo =".".join(resultOfClingo) + "."
            text_file = open(resultLocation, "w")
            text_file.write(resultOfClingo[1:]+ "".join(MultipleTimesOccuringConflicts))
            
            text_file.close()
            
        else:
            print("Edge, stopped after iteration " + str(i))
            MultipleTimesOccuringConflicts.clear()
            break

#change here the last element to change conflict solving method
inputs = ["", "encodings/singleAgentPF.lp","encodings/conflict_detection.lp","encodings/collision_evasion.lp","encodings/collision-avoidance-wait.lp","encodings/collisionToRPosition.lp","encodings/rPositionToCollision.lp"]

for i in range(1,len(sys.argv)):
    inputs[i-1] = sys.argv[i]

print(inputs)
#list containing all generated paths
paths = []

#number of how many robots the instance has
numberOfRobots = 30
for i in range(1,numberOfRobots +1):
    
    c = clingo.clingo_main(Application(sys.argv[0]), [inputs[0] , inputs[1] ,"--outf=3", "-c horizon=40", "-c id ="+ str(i)])
    paths.append(resultOfClingo)

#joins and cleans all the paths
pathsClean = remove_duplicates("".join(paths))


instanceName = path_leaf(str(inputs[0]))[0:-3]
#creates directory with the name of the instance
try:
    mkdir(instanceName)
except: pass

#set location for the output files


pathLocation = "./"+instanceName+"/Paths-"+instanceName+ ".lp"
colissionLocation = "./"+instanceName+"/Collisions-"+instanceName+ ".lp"
resultLocation = "./"+instanceName+"/NewPlan-"+instanceName+ ".lp"

#saves joined paths
text_file = open(pathLocation, "w")
text_file.write(pathsClean)
text_file.close()

#creates and saves all conflicts
c = clingo.clingo_main(Application(sys.argv[0]), [inputs[5],pathLocation ,"--outf=3"])
text_file = open(resultLocation, "w")
text_file.write(resultOfClingo)
text_file.close()

# if "edge" in resultOfClingo:

#     #creates and saves the found solution
#     c = clingo.clingo_main(Application(sys.argv[0]), [inputs[3],colissionLocation ,"--outf=3"])
#     resultOfClingo = resultOfClingo.split(".")
#     resultOfClingo.sort()
#     resultOfClingo =".".join(resultOfClingo) + "."
#     text_file = open(resultLocation, "w")
#     text_file.write(resultOfClingo[1:])
#     text_file.close()



solveEdge()


for i in range(0,vertexIterations):

    #creates and saves all conflicts
    c = clingo.clingo_main(Application(sys.argv[0]), [inputs[2],resultLocation ,"--outf=3"])
    text_file = open(colissionLocation, "w")
    text_file.write(resultOfClingo)
    text_file.close()

    skipvertext = False
    for j in resultOfClingo.split("."):
        if "edgeCollision(" in j:
            if not "edgeCollision(1" in j and edgeIterations != 0:
                print("Found edge")
                skipvertext = True
                solveEdge()
        if skipvertext:
            skipvertext = False
            continue

    if "ver" in resultOfClingo:
        

        
        

            #creates and saves the found solution
        c = clingo.clingo_main(Application(sys.argv[0]), [inputs[4],colissionLocation ,"--outf=3"])
        print("\n\nVertex - Iteration " + str(i)+ "\n")
        if "rPositionX" in resultOfClingo:
            print("No solution in vertex, iteration" + str(i))
            print(resultOfClingo)
            break
        
        
        
        for j in resultOfClingo.split("."):
            if "ver" in j:
                print(j)
            if "hoice" in j:
                print(j)
        text_file = open(resultLocation, "w")
        text_file.write(resultOfClingo)
        text_file.close()
        
    else:
        print("Vertex. stopped after iteration " + str(i))
        break

#creates and saves all conflicts
c = clingo.clingo_main(Application(sys.argv[0]), [inputs[6],resultLocation ,"--outf=3"])

resultOfClingo = resultOfClingo.split(".")
resultOfClingo.sort()
resultOfClingo =".".join(resultOfClingo) + "."

text_file = open(resultLocation, "w")
text_file.write(resultOfClingo[1:])
text_file.close()

print("\n\n Starting Visualizer \n\n")
system('viz -p ' + resultLocation)








