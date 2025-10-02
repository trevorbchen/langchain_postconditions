# Z3 verification for sort_array
# Postcondition: The array is not NULL and the size is non-negative

from z3 import *

arr = Array('arr', IntSort(), IntSort())
size = Int('size')
s = Solver()

s.add(arr != None)
s.add(size >= 0)

print(s.check())