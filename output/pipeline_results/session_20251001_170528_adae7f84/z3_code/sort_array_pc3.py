# Z3 verification for sort_array
# Postcondition: The count of each element in the sorted array is the same as in the original array

from z3 import *

arr = Array('arr', IntSort(), IntSort())
original_arr = Array('original_arr', IntSort(), IntSort())
size = Int('size')
s = Solver()

for i in range(size):
    s.add(arr.count(i) == original_arr.count(i))

print(s.check())