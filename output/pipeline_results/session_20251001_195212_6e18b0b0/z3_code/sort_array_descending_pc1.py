# Z3 verification for sort_array_descending
# Postcondition: Array is sorted in non-increasing order

from z3 import *

arr = Array('arr', IntSort(), IntSort())
size = Int('size')
s = Solver()

i = Int('i')
j = Int('j')
s.add(ForAll([i, j], Implies(And(0 <= i, i < j, j < size), arr[i] >= arr[j])))

print(s.check())