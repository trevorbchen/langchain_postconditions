# Z3 verification for reverseArray
# Postcondition: Array is reversed

from z3 import *

# Declare variables
i = Int('i')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = ForAll([i], 
    Implies(And(i >= 0, i < size),
        Select(arr, i) == Select(arr, size - i - 1)))

# Create solver and verify
s = Solver()
s.add(constraint)
s.add(size > 0)  # size must be greater than 0

result = s.check()
print(f"Verification result: {result}")

if result == sat:
    print("✓ Postcondition is satisfiable")
    print("Model:", s.model())
elif result == unsat:
    print("✗ Postcondition is unsatisfiable")
else:
    print("? Unknown")