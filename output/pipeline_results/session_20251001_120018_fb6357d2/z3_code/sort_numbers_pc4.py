# Z3 verification for sort_numbers
# Postcondition: The size of the array is non-negative

# ======================================================================
# ✅ VALIDATION PASSED
# ======================================================================
# Status: GREEN
# Code is syntactically correct and well-formed
# Validation Status: success
#
# Declared Variables:
#   - x: Int
#   - size: Int
#   - arr: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
x = Int('x')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')
# Define constraint
constraint = And(size >= 0, ForAll([x], Implies(And(x >= 0, x < size), Select(arr, x) <= Select(arr, x+1))))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")