import sys
import cu_matrix_mmul

import numpy as np

A = np.random.randint(10, size=(2,3))
B = np.random.randint(10, size=(3,3))

print(A)
print(B)

C = cu_matrix_mmul.mmul(A, B)
print(C)