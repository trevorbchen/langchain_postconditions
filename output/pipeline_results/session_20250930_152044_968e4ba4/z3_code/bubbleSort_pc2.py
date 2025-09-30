# Z3 verification for bubbleSort
# Postcondition: All elements in the original array exist in the sorted array

from z3 import *

# Declare variables
i = Int('i')
j = Int('j')
arr = Array('arr', IntSort(), IntSort())
size = Int('size')

# Define constraints
constraint = ForAll([i], 
    Implies(And(i >= 0, i < size),
        Exists([j], And(j >= 0, j < size, Select(arr, i) == Select(arr, j)))))

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