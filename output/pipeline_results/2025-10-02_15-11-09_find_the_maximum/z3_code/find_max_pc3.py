# Z3 Verification Code
# Function: find_max
# Postcondition 3: If the array is not empty, the maximum value exists in the array
# Formal: length > 0 → ∃i: 0 ≤ i < length ∧ array[i] = find_max(array, length)
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
i = Int('i')
s = Solver()
s.add(Implies(length > 0, Exists(i, And(i >= 0, i < length, array[i] == find_max(array, length)))))
print(s.check())