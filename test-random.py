from random123 import solver
v = [5, 8, 3, 2, 8, 4]
weight = [3, 5, 2, 1, 4, 8]
W = 12

solution = solver(v, weight, W).solve()
print(solution)