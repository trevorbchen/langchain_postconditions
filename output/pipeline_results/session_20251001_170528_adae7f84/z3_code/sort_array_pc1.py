# Z3 verification for sort_array
# Postcondition: Array is sorted in non-decreasing order

from z3 import *

arr = Array('arr', IntSort(), IntSort())
size = Int('size')
s = Solver()

for i in range(size):
    for j in range(i+1, size):
        s.add(arr[i] <= arr[j])

print(s.check())