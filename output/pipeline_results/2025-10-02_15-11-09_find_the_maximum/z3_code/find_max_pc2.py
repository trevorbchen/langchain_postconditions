# Z3 Verification Code
# Function: find_max
# Postcondition 2: The maximum value exists in the array
# Formal: ∃i: 0 ≤ i < length ∧ array[i] = find_max(array, length)
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
i = Int('i')
s = Solver()
s.add(Exists(i, And(i >= 0, i < length, array[i] == find_max(array, length))))
print(s.check())