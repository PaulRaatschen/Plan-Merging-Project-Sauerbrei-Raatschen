import subprocess
import sys
from os import path

WORKING_DIR : str = path.abspath(path.dirname(__file__))


#subprocess.run('python naiveplanning/SequentialPlanner.py naiveplanning/instances/9r_singleOpening.lp 0 0',shell =True)

p1 = subprocess.Popen(['python', path.join(WORKING_DIR,'naiveplanning/SequentialPlanner.py'),'naiveplanning/instances/9r_singleOpening.lp','76','76'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p2 = subprocess.Popen(['python', path.join(WORKING_DIR,'prioritizedPlanning/prioritized_planning.py'),'naiveplanning/instances/9r_singleOpening.lp','-b'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("\n\n")
out, err = p1.communicate()
print(out)
print(err)
print("\n\n")
out, err = p2.communicate()
print(out)
print(err)
