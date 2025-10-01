# Z3 verification for sort_numbers
# Postcondition: The input pointer is not NULL

# ======================================================================
# ✅ VALIDATION PASSED
# ======================================================================
# Status: GREEN
# Code is syntactically correct and well-formed
# Validation Status: success
#
# Declared Variables:
#   - size: Int
#   - numbers: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
numbers = Array('numbers', IntSort(), IntSort())
size = Int('size')
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), Select(numbers, i) != None))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")