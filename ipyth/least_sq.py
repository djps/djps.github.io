import numpy as np
from scipy.optimize import least_squares

def p(beta, u):
    return beta[0] * np.cos(u) + beta[1] * u
	
u_i = np.array([0, np.pi /4.0, np.pi / 2.0])
p_i = np.array([5.0, 4.0, 6.0])

def fun(beta):
    return p(beta, u_i) - p_i
	
beta0 = np.array([1.0, 1.0])

# compute the least squares solution
res = least_squares(fun, beta0)

print("beta:", res.x)
# Note https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares
print("cost:", 2.0 * res.cost)
print("fun:", res.fun)
print("message:", res.message)
print("success:", res.success)

# analytical results
a = (25.0 + 2.0 * np.sqrt(2.0)) / 7.0
b = (88.0 - 10.0 * np.sqrt(2.0)) / (7.0*np.pi) 

print("a:", (25.0 + 2.0*np.sqrt(2.0)) / 7.0)
print("b:", (88.0 - 10.0 * np.sqrt(2.0)) / (7.0*np.pi) )

E = (p_i[0] - p([a,b], u_i)[0])**2 + (p_i[1] - p([a,b], u_i)[1])**2 + (p_i[2] - p([a,b], u_i)[2])**2

print(E)