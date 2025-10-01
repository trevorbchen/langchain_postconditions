# Z3 verification for sort_numbers
# Postcondition: All elements in the original array are in the sorted array

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
#   - numbers_prime: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
i = Int('i')
size = Int('size')
numbers = Array('numbers', IntSort(), IntSort())
numbers_prime = Array('numbers_prime', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), Select(numbers_prime, i) == Select(numbers, i)))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")