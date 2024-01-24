import numpy as np
import matplotlib.pyplot as plt
import copy

## Nonlinear solvers: Newton method vs bisection method

# One needs the derivative of the original function for Newton method. Here it is given analytically.

# It can also be estimated with central or other differencing schemes (e.g. Secant method).

# For Newton method to converge, we need to be 'sufficiently' close to the root of the nonlinear equation and the function needs to be sufficiently smooth (differentiable; actually in practice often twice differentiable). Under certain conditions, the convergence is quadratic.

# For bisection method we only need a continuous function and a change of sign between the endpoints of the interval. However, convergence is only linear.

# Take the function $x^3$ first. Analytic root is $0$. Our starting point for Newton method is $1$.

# Function is x^3
def f(x):
    v=x**3
    return v

# Derivative of function, here analytically
def f_deriv(x):
    dv=3*x**2
    return dv

# Array to save iteration results
x_N1=np.zeros(20)

# Initial guess
x_N1[0]=1.

# Newton method with 20 iterations
for i in np.arange(1,20):
    x_N1[i] = x_N1[i-1] - f(x_N1[i-1]) / f_deriv(x_N1[i-1])

# Plotting solution; analytic  root should be 0
plt.plot(x_N1)
plt.xlim([0,20])
plt.grid()

# In this case, convergence is relatively slow. One can observe a saddle point of the function at the root $0$ which causes the slower convergence. Shown below is the error with respect to the analytical solution for each iteration.

# Plotting of iterations
for i in np.arange(0,20):
    print('%.52f' % np.abs(x_N1[i]-0))

# Next, we try the function $x^3-2$. We use both bisection method and Newton method. Initial interval is $[-2,2]$ and initial guess for Newton method is 2. Analytic solution is $2^{1/3}$.

# Function is x^3-2
def f(x):
    v = x**3 - 2
    return v

# Derivative of function, here analytically
def f_deriv(x):
    dv = 3 * x**2
    return dv

# Array to save iteration results
x_B=np.zeros(20)

#Endpoints of initial interval for bisection method.
x_B[0]=-2.
x_B[1]=2.
a = -2.
b = 2.

# Bisection method; One needs to first check that it works (i.e. sign change).
if(f(a)*f(b) < 0):
    for i in np.arange(2,20):
        # Bisect
        x_B[i] = (b + a) / 2.0
        # Pick the right half of the interval, so that a sign change occurs
        if (f(x_B[i]))*(f(a))<0:
            b=x_B[i]
        else:
            a=x_B[i]
        #Break off, if we found the solution; in practice, a sufficiently close solution would also work
        #e.g., np.abs(f(x_B[i]))<10**-7 or something like that
        #or the difference between two new iterations is very small,
        #i.e. the update from one iteration to the next is smaller than some threshold
        if (f(x_B[i]))==0:
            print('Found root')
            break
else:
    print('No sign change; bisection does not work!')

# Netwon method; same as before:
x_N = np.zeros(20)
x_N[0] = 2.0
for i in np.arange(1,20):
    x_N[i]=x_N[i-1]-(f(x_N[i-1]))/(f_deriv(x_N[i-1]))

# Plotting iterations for both methods (orange: Newton, blue: bisection)
plt.plot(x_B, label='Bisection')
plt.plot(x_N, label='Newton')
plt.legend()
plt.xlim([0,20])
plt.grid()


# Error of bisection method for each iteration. One can see a linear reduction of error. Sometimes error gets larger again, as the selection of the interval changes (left or right part of bisected interval).

for i in np.arange(0,20):
    print('%.52f' % np.abs(x_B[i]-x_true))


# Error of Newton method for each iteration. One can see a quadratic reduction of error. As the error gets smaller, especially below $1$, the amount of correct digits suddenly increases drastically from one iteration to the next. We need only 7 iterations (excluding the initial guess) to reach machine precision.

for i in np.arange(0,20):
    print('%.52f' % np.abs(x_N[i]-x_true))


# # Nonlinear solver, 2D-Newton method (can be extended to any higher dimension)

# Application of the Newton Method to the function $f(x,y)=(x^2-64, x+y^3)^T$. One needs the Jacobian matrix (i.e. matrix of partial derivatives) for this. Here it is given analytically.

# Similar as for the 1D case, it can also be estimated with central or other differencing schemes.

# For Newton method to converge, we need to be 'sufficiently' close to the root of the nonlinear equation and the function needs to be sufficiently smooth (differentiable).

# If it converges and certain conditions hold, the convergence is quadratic.

#F unction (x^2-64, x+y^3)
def f(x):
    v=[x[0]**2-64.,x[0]+x[1]**3]
    return v

# Derivative of function, here analytically
def f_deriv(x):
    dv=[[2*x[0], 0],[1,3.*x[1]**2]]
    return dv

# Starting point and iteration
x=np.array([1.,-2.])
x_o=np.array([0.,0.])
i=1

# Do Newton iteration until the difference between successive steps is <=10^-4
while (np.linalg.norm(x-x_o)>10**-4):
    x_o=x
    fu=f(x)
    dfu=f_deriv(x)
    in_dfu=np.linalg.inv(dfu)
    x=x-in_dfu.dot(fu)
    i+=1
print('Solution estimate is ', x, ' after ', i, ' iterations ' )
