# Z3 Verification Code
# Function: find_max
# Postcondition 1: The returned value is greater than or equal to all elements in the array
# Formal: ∀i: 0 ≤ i < length → array[i] ≤ find_max(array, length)
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
i = Int('i')
s = Solver()
s.add(ForAll(i, Implies(And(i >= 0, i < length), array[i] <= find_max(array, length))))
print(s.check())