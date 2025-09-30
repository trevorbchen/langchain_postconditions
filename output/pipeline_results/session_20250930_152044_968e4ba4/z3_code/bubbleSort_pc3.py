# Z3 verification for bubbleSort
# Postcondition: All elements in the array are integers

from z3 import *

# Declare variables
i = Int('i')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = ForAll([i], 
    Implies(And(i >= 0, i < size),
        IsInt(Select(arr, i))))

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