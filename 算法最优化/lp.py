import numpy as np
from cvxopt import matrix, solvers

A = matrix([[2.0,-2.0], [3.0,-3.0],[-5.0,5.0]])
b = matrix([1.0,1.0])
c = matrix([-1.0, 1.0,0.0])

sol = solvers.lp(c,A,b)

print(sol['x'])
print(np.dot(sol['x'].T, c))
print(sol['primal objective'])