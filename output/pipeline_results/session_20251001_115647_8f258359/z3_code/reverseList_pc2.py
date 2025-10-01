# Z3 verification for reverseList
# Postcondition: No element in the list is NULL

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
#   - list: Array
#
# Declared Sorts: Int, Array
# ======================================================================

from z3 import *
# Declare variables
i = Int('i')
size = Int('size')
list = Array('list', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), list[i] != None))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")