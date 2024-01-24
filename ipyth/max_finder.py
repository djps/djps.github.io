import math
import numpy as np
from matplotlib import pyplot as plt


# We need to find maximum of this $g(x)$
# \begin{equation}
# e^{x^2 + 1} - 2.718 + 0.16969 x - 4.07799 x^2 + 3.3653 x^3 - 4.1279 x^4
# \end{equation}
#
# To find maximum of $g(x)$, we need to find roots of $g^{\prime}(x)$, so a function `gprime`
#
# To find roots and confirm if it's a maxima, we need `gdoubleprime(x)`, as $g(x)$ is at maximum if $g^{\prime\prime}(x)$ is negative


def g(x):
    return np.exp(x**2 + 1) - 2.718 + 0.16969*x - 4.07799*x**2 + 3.3653 * x**3 - 4.1279 * x**4

def gprime(x):
    return np.exp(x**2 + 1) * 2 * x + 0.16969 - 2.0 * 4.07799 * x + 3 * 3.3653 * x**2 - 4 * 4.1279 * x**3

def gdoubleprime(x):
    return 2 * (2 * x + 1) * np.exp(x**2 + 1) - 2.0 * 4.07799 + 3 * 2 * 3.3653 * x - 4 * 3 * 4.1279 * x**2

x = np.linspace(0, 1, 100)
plt.plot(x, g(x))
plt.grid(True)


# Newton method
def nm_step(x, step_num):
    val = x - gprime(x) / gdoubleprime(x)
    step_num += 1
    print(f"Step Num: {step_num}, x: {val}")
    return val, step_num

# Secant method

def secant_step(x1, x2, step_num, f):
    step_num += 1
    val = x2 - f(x2)*(x2 - x1) / (f(x2) - f(x1))
    print(f"Step Num: {step_num}, x: {val}")
    return x2, val, step_num


# Execution: We use secant method here because we can bound it.


def calculate_first_max():
    step_num = 0
    x1 = 0      # These bounds are found by analyzing the graph
    x2 = 0.2
    for i in range(19):
        x1, x2, step_num = secant_step(x1, x2, step_num, gprime)
    return x2

def calculate_second_max():
    step_num = 0
    x1 = 0.6      # These bounds are found by analyzing the graph
    x2 = 0.7
    for i in range(6):
        x1, x2, step_num = secant_step(x1, x2, step_num, gprime)
    return x2

print("Calculating x for first maximum in x -> [0, 1], in interval [0, 0.2]:")
max_x_1 = calculate_first_max()
print("First maximum value: ", g(max_x_1))

print("Calculating x for second maximum in x -> [0, 1], in interval [0.6, 0.7]:")
max_x_2 = calculate_second_max()
print("Second maximum value: ", g(max_x_2))

plt.plot(x, g(x))
plt.plot(max_x_1, g(max_x_1), '*')
plt.plot(max_x_2, g(max_x_2), 'o')
plt.grid(True)
