#   #################################################
#   compuational science project 3.1
#   simulation of motion of planets around the sun

#   Author - Girmay Asrat,  March 2021
#   #################################################

import sys
import numpy as np
import math
from matplotlib import pyplot as plt 
from prettytable import PrettyTable
import docx 


if  len (sys.argv ) != 2 :
    print("usage is : %s input[text file] " % (sys.argv[0])) 
    sys.exit()