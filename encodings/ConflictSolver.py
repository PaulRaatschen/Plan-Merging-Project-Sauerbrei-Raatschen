"""
ConflictSolver.py

Input: File becomes 4 clingo files as input: "instance.lp" singleAgentPF.lp conflict_detection.lp "conflictsolving.lp"
example in Anaconda:

python ConflictSolver.py Cross_2rB.lp singleAgentPF.lp conflict_detection.lp collision-avoidance.lp

Output: Creates a new directory, named after the instance,  containing a file which contains all paths,
                                                            containing a file with the conflict model
                                                            & containing a file with the new path solution.

Needs to run in an environment which supports clingo applications
"""


import sys #used to give .lp files in the prompt
import ntpath
import clingo
from os import mkdir #to create a folder which saves the results
from collections import Counter #used toremove duplicates

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

#list containing all generated paths
paths = []

#number of how many robots the instance has
numberOfRobots = 4
for i in range(1,numberOfRobots +1):
    
    c = clingo.clingo_main(Application(sys.argv[0]), [sys.argv[1] , sys.argv[2] ,"--outf=3", "-c horizon=20", "-c id ="+ str(i)])
    paths.append(resultOfClingo)

#joins and cleans all the paths
pathsClean = remove_duplicates("".join(paths))


instanceName = path_leaf(str(sys.argv[1]))[0:-3]
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
c = clingo.clingo_main(Application(sys.argv[0]), [sys.argv[3],pathLocation ,"--outf=3"])
text_file = open(colissionLocation, "w")
text_file.write(resultOfClingo)
text_file.close()

#creates and saves the found solution
c = clingo.clingo_main(Application(sys.argv[0]), [sys.argv[4],colissionLocation ,"--outf=3"])
resultOfClingo = resultOfClingo.split(".")
resultOfClingo.sort()
resultOfClingo =".".join(resultOfClingo) + "."
text_file = open(resultLocation, "w")
text_file.write(resultOfClingo[1:])
text_file.close()
