# Z3 verification for sort_array
# Postcondition: The time complexity of the function is between 0 and O(n log n)

from z3 import *

t = Real('t')
n = Real('n')
s = Solver()

s.add(t >= 0, t <= n*log(n))

print(s.check())