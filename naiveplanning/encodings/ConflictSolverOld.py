"""
ConflictSolver.py
Input:  Instance you want to run
example in Anaconda:
python encodings\ConflictSolver.py instances\Cross_2rB.lp 
Output: Creates a new directory, named after the instance,  containing a file which contains all paths,
                                                            containing a file with the latest conflict model
                                                            & containing a file with the new path solution.
Needs to run in an environment which supports clingo applications
"""


import sys #used for accessing the given instance
import ntpath
import clingo #used to run the clingo parts of the encoding
from os import mkdir #used to create a new folder, which saves the solution
from os import system #used to run command lines in the encodings, here only opening the asprillo visualiser
from collections import Counter #used to modify Strings, here to remove duplicate entries from generated .lp files




edgeIterations = 500 #number of run edge iterations

vertexIterations = 70 #number of run vertext iterations

numberOfRobots = 30 #max number of how many robots the instance has, can be higher then the total amount

#standard clingo function for running .lp files
#saves solution into global string resultOfClingo
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
        
        #Solves and saves result into global variable
        with ctl.solve(yield_=True) as handle:
            
            global resultOfClingo
            for m in handle: resultOfClingo = format(m).replace(" ", ".") + "."

#removes duplicate inputs, here used to unite all individual planned paths at the beginning
def remove_duplicates(input):
 
    #split input string separated by "."
    input = input.split(".")
 
    #create dictionary consisting of unique entries
    UniqW = Counter(input)
 
    #joins unique entries back together
    s = ".".join(UniqW.keys())
    return(s)

#function which gets the name of the running instance, used for naming the resulting directory
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

#function solving the edge collisions in a .lp file, over multiple iterations
def solveEdge():

    global edgeIterations
    global maxEdgeIterations
    
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

        #if an edge collision was found
        if "edge" in resultOfClingo:

            #solve edge collision
            c = clingo.clingo_main(Application(sys.argv[0]), [inputs[3],colissionLocation ,"--outf=3"])
            
            #if edge collision failed (if it failed, resultOfClingo would contain the term edgeCollision, otherwise not)
            if "edge" in resultOfClingo:

                print("No solution in itearation " + str(maxEdgeIterations - edgeIterations))
                print("Collisions at that point were:")
                
                for j in resultOfClingo.split("."):
                    if "Col" in j:
                        print(j + "\n")
                
                #end function if edge couldn't be solved, in hope that vertex would solve it
                return

            #if edge collision was found
            print("\n\nEdge - Iteration " + str(maxEdgeIterations - edgeIterations)+ "\n")
            for j in resultOfClingo.split("."):
                
                if "conflict" in j:
                    
                    #check if an previos iteration solved it -> if yes then there would be a loop in the function
                    if j in ListOfConflicts:
                        print("Conflict found in a previos iteration, now removed from further iterations, until all other edge conflicts are solved first")
                        MultipleTimesOccuringConflicts.append(j+".")
                    #add conflict to already solved 
                    ListOfConflicts.append(j)
                    #print the conflict
                    print(j + "\n")

                #print how long one of the robots waited
                if "wait" in j:
                    print(j)

            #save result
            text_file = open(resultLocation, "w")
            text_file.write(resultOfClingo[1:]+ "".join(MultipleTimesOccuringConflicts))
            
            text_file.close()

            #repeat
        
        #if no edge collision was found, end edge iteration
        else:
            print("Edge, stopped after iteration " + str(maxEdgeIterations - edgeIterations))

            #free list of multiple time occuring conflicts
            MultipleTimesOccuringConflicts.clear()
            return


#list of all solved conflicts -> only used for edge collision
ListOfConflicts = []

#list of conflicts, apperearing more then one time
MultipleTimesOccuringConflicts = []

#String saving the result of the clingo function
resultOfClingo = ""

maxEdgeIterations = edgeIterations
maxVertexdgeIterations = vertexIterations


inputs = ["", "encodings/singleAgentPF.lp","encodings/conflict_detection.lp","encodings/collision_evasion.lp","encodings/collision-avoidance-wait.lp","encodings/collisionToRPosition.lp","encodings/rPositionToCollision.lp"]

#uses inputs as files to solve .lp files, now you should only enter the instance
for i in range(1,len(sys.argv)):
    inputs[i-1] = sys.argv[i]

#list the used encodings
print(inputs)

#list containing all generated paths
paths = []

#generating all individual paths, using singleAgentPF.lp
for i in range(1,numberOfRobots +1):
    c = clingo.clingo_main(Application(sys.argv[0]), [inputs[0] , inputs[1] ,"--outf=3", "-c horizon=40", "-c id ="+ str(i)])
    paths.append(resultOfClingo)

#joins and cleans all the individual paths
pathsClean = remove_duplicates("".join(paths))

#finds name of the instance, example: instance1.lp -> instance1
instanceName = path_leaf(str(inputs[0]))[0:-3]

#creates directory with the name of the instance
try:
    mkdir(instanceName)
except: pass

#set location for the output files
pathLocation = "./"+instanceName+"/Paths-"+instanceName+ ".lp"
colissionLocation = "./"+instanceName+"/Collisions-"+instanceName+ ".lp"
resultLocation = "./"+instanceName+"/NewPlan-"+instanceName+ ".lp"

#saves the joined paths
text_file = open(pathLocation, "w")
text_file.write(pathsClean)
text_file.close()

#transforms indiviual paths into paths represented by object rPosition
c = clingo.clingo_main(Application(sys.argv[0]), [inputs[5],pathLocation ,"--outf=3"])
text_file = open(resultLocation, "w")
text_file.write(resultOfClingo)
text_file.close()

solveEdge()


#solves vertex collisions, when it creates edge collision, then solved them first
for i in range(1,vertexIterations+1):

    #creates and saves all conflicts
    c = clingo.clingo_main(Application(sys.argv[0]), [inputs[2],resultLocation ,"--outf=3"])
    text_file = open(colissionLocation, "w")
    text_file.write(resultOfClingo)
    text_file.close()

    #variable to say if vertex solving should be skipped due to edge running prio
    skipvertex = False

    for j in resultOfClingo.split("."):
        #if edge collision gets found
        if "edgeCollision(" in j:
            #not placeholder edge collision and program allowed to solve more edge collisions
            if not "edgeCollision(1" in j and edgeIterations != 0:
                print("Found edge during vertex collision, solving it")
                skipvertex = True
                solveEdge()
        if skipvertex:
            skipvertex = False
            continue

    #if vertex collision was found
    if "vertex" in resultOfClingo:
        

        
        

        #creates and saves the found solution
        c = clingo.clingo_main(Application(sys.argv[0]), [inputs[4],colissionLocation ,"--outf=3"])
        print("\n\nVertex - Iteration " + str(i)+ "\n")
        
        #if the vertex collision file failed
        if "rPositionX" in resultOfClingo:
            print("No solution in vertex, iteration" + str(i))
            print(resultOfClingo)
            break
        
        
        
        for j in resultOfClingo.split("."):
            #print solved vertex collision
            if "first" in j:
                print(j)
            #print choice, which robot waited
            if "hoice" in j:
                print(j)
        
        #save all
        text_file = open(resultLocation, "w")
        text_file.write(resultOfClingo)
        text_file.close()
        
    else:
        print("Vertex. stopped after iteration " + str(i))
        break

#rewrite rPositions back into occurs
c = clingo.clingo_main(Application(sys.argv[0]), [inputs[6],resultLocation ,"--outf=3"])

#sort result so visualizer won't bug
resultOfClingo = resultOfClingo.split(".")
resultOfClingo.sort()
resultOfClingo =".".join(resultOfClingo) + "."

#save result
text_file = open(resultLocation, "w")
text_file.write(resultOfClingo[1:])
text_file.close()

#start the visualizer
print("\n\n Starting Visualizer \n\n")
system('viz -p ' + resultLocation)








