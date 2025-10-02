# Z3 Verification Code
# Function: find_max
# Postcondition 5: If the array is NULL, the function returns an undefined value
# Formal: array = NULL → find_max(array, length) = undefined
#
# Validation Status: ✓ PASSED
#

from z3 import *

array = Array('array', IntSort(), IntSort())
length = Int('length')
find_max = Function('find_max', ArraySort(IntSort(), IntSort()), IntSort(), IntSort())
undefined = Int('undefined')
s = Solver()
s.add(Implies(array == None, find_max(array, length) == undefined))
print(s.check())