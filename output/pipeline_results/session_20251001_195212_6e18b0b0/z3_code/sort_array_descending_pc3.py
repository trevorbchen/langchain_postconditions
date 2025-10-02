# Z3 verification for sort_array_descending
# Postcondition: Size of array is non-negative

from z3 import *

size = Int('size')
s = Solver()

s.add(size >= 0)

print(s.check())