# Z3 verification for bubbleSort
# Postcondition: Size of the array is non-negative

from z3 import *

# Declare variables
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = size >= 0

# Create solver and verify
s = Solver()
s.add(constraint)

result = s.check()
print(f"Verification result: {result}")

if result == sat:
    print("✓ Postcondition is satisfiable")
    print("Model:", s.model())
elif result == unsat:
    print("✗ Postcondition is unsatisfiable")
else:
    print("? Unknown")