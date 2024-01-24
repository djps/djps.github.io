""""
The order of operations and the respective number representation can lead to rounding errors and propagation of errors.

The order of operations matters on computers as addition/subtraction is not associative.

Here, addition of small number $b$ to a large number $a$ can cause a problem.

In a python environment, import the libraries we need:
"""

import numpy as np
import math

# Here c and d should be identical:

a = 10**7
b = 10**(-20)
c = (a + b - a)**(1/10.0)
d = (a - a + b)**(1/10.0)
print('Solution 1 is ', c,' and Solution 2 is ', d)

# Subtraction of values of similar size also leads to loss of precision (with double precision often quite difficult, as there may actually be extra digits available for certain computations). But it still becomes obvious for ${x = 1/150000000}$ in the example.

# Alternative application: use an approximation of sine by the Taylor expansion for angles close to zero to yield a better answer, because first terms of Taylor expansion are accurate enough and we work with numbers of not too similar and not too different sizes in the expansion.

# Try also removing some zeros below

x = 1.0 / 150000000.0
y = math.sin(x)
z = x - y
print('%.90f' % x)
print('%.90f' % y)
print('%.90f' % z)

# If we use the Taylor series approximation of $x-\sin(x)$ for $x$ close to $0$; we can get even better by adding more terms

z = x - x + x**3 / np.math.factorial(3) - x**5 / np.math.factorial(5) + x**7 / np.math.factorial(7)

print('%.90f' % z)
