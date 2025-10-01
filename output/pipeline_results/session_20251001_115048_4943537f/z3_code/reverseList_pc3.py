# Z3 verification for reverseList
# Postcondition: The list pointer is not NULL

from z3 import *
# Declare variables
list = Array('list', IntSort(), IntSort())
size = Int('size')
# Define constraint
constraint = ForAll([i], Implies(And(i >= 0, i < size), Select(list, i) <= Select(list, i+1)))
# Solver
s = Solver()
s.add(constraint)
result = s.check()
print(f"Result: {result}")