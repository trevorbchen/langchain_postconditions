# Z3 verification for sort_numbers
# Postcondition: Array is sorted in ascending order

# ======================================================================
# ✅ VALIDATION PASSED
# ======================================================================
# Status: GREEN
# Code is syntactically correct and well-formed
# Validation Status: success
#
# Declared Variables:
#   - i: Int
#   - j: Int
#   - size: Int
#   - numbers: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
i = Int('i')
j = Int('j')
size = Int('size')
numbers = Array('numbers', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i, j], Implies(And(i >= 0, j > i, j < size), Select(numbers, i) <= Select(numbers, j)))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")