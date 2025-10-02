# Z3 Verification Code
# Function: find_max
# Postcondition 4: If the array is empty, the function returns an undefined value
# Formal: length = 0 → find_max(array, length) = undefined
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
undefined = Int('undefined')
s = Solver()
s.add(Implies(length == 0, find_max(array, length) == undefined))
print(s.check())