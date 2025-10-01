# Z3 verification for sort_numbers
# Postcondition: All elements in the array are integers

# ======================================================================
# ✅ VALIDATION PASSED
# ======================================================================
# Status: GREEN
# Code is syntactically correct and well-formed
# Validation Status: success
#
# Declared Variables:
#   - i: Int
#   - size: Int
#   - numbers: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
i = Int('i')
size = Int('size')
numbers = Array('numbers', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), numbers[i] == ToInt(numbers[i])))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")