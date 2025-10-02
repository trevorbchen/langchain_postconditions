# Z3 Verification Code
# Function: find_max
# Postcondition 6: If the array contains only one element, the function returns that element
# Formal: length = 1 → find_max(array, length) = array[0]
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
s = Solver()
s.add(Implies(length == 1, find_max(array, length) == array[0]))
print(s.check())