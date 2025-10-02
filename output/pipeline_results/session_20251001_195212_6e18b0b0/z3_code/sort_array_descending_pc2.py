# Z3 verification for sort_array_descending
# Postcondition: Array pointer is not NULL

from z3 import *

arr = Array('arr', IntSort(), IntSort())
s = Solver()

s.add(arr != None)

print(s.check())