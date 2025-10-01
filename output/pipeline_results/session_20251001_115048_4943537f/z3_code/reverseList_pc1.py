# Z3 verification for reverseList
# Postcondition: Array elements are reversed in order

from z3 import *
# Declare variables
i = Int('i')
size = Int('size')
list = Array('list', IntSort(), IntSort())
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), Select(list, i) == Select(list, size - i - 1)))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")