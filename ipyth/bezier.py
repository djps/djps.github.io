## Bezier Curves

# An example on Bezier curves

import numpy as np
import matplotlib.pyplot as plt


# Bezier formulas for $x$ and $y$ axes

def bx(t):
    return 3.0 * t

def by(t):
    return 2.0 + 3.0 * t - 12.0 * t**2 + 7.0 * t**3

# Provided $x$ and $y$ values

x = [0, 1, 2, 3]
y = [2, 3, 0, 0]

# Evaluating over the discretized interval with 100 points

n: int = 100
u = np.linspace(np.min(x), np.max(x), n)

# Normalize to interval [0, 1]

t = (u - np.min(x)) / np.max(x)

# Calculate Bezier values for both axes

bxt = [bx(i) for i in t]
byt = [by(i) for i in t]


# Now plot

plt.plot(x, y, 'ro--', label = ' Points')
plt.plot(bxt, byt, 'b-', label = 'Bezier Curve')
plt.legend()
plt.grid()
plt.show()


# In[ ]:
