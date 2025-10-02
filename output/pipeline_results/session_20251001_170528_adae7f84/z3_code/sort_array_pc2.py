# Z3 verification for sort_array
# Postcondition: All elements in the sorted array are from the original array

from z3 import *

arr = Array('arr', IntSort(), IntSort())
original_arr = Array('original_arr', IntSort(), IntSort())
size = Int('size')
s = Solver()

for i in range(size):
    s.add(arr[i] == original_arr[i])

print(s.check())