# Z3 verification for sort_numbers
# Postcondition: The size of the array is within the integer limit

# ======================================================================
# ✅ VALIDATION PASSED
# ======================================================================
# Status: GREEN
# Code is syntactically correct and well-formed
# Validation Status: success
#
# Declared Variables:
#   - size: Int
#
# Declared Sorts: Int
# ======================================================================

from z3 import *
# Declare variables
size = Int('size')
# Define constraint
constraint = size <= IntVal(2147483647)
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")