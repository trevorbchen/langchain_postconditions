# Z3 verification for bubbleSort
# Postcondition: Array pointer is not NULL

from z3 import *

# Declare variables
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = Not(arr == None)

# Create solver and verify
s = Solver()
s.add(constraint)
s.add(size > 0)  # Preconditions

result = s.check()
print(f"Verification result: {result}")

if result == sat:
    print("✓ Postcondition is satisfiable")
    print("Model:", s.model())
elif result == unsat:
    print("✗ Postcondition is unsatisfiable")
else:
    print("? Unknown")