import numpy as np
from math import factorial 
import matplotlib.pyplot as plt

# x values
x = np.linspace(-np.pi, np.pi, 200) 

# setup figure
plt.figure(figsize = (10,8))

# First order approximation
n = 0
# approximate y values
y = np.zeros(len(x))
# Because only every other term is not equal to zero, we only have to go to n/2
n = n / 2
# Taylor series expansion up to n
for i in np.arange(0, int(n)+1):
    y = y + ((-1)**i * (x)**(2*i+1)) / factorial(2*i+1)
# Plotting    
plt.plot(x,y, label = 'First Order')

# Third order (same structure as before)
n = 2
y = np.zeros(len(x))
n = n / 2
for i in np.arange(0, int(n+1)):
    y = y + ((-1)**i * (x)**(2*i+1)) / factorial(2*i+1)
plt.plot(x, y, label = 'Third Order')

# Seventh order
n = 6
y = np.zeros(len(x))
n = n / 2
for i in np.arange(0,int(n+1)):
    y = y + ((-1)**i * (x)**(2*i+1)) / factorial(2*i+1)
plt.plot(x, y, label = 'Seventh Order')

# Eleventh order
n = 10
y = np.zeros(len(x))
n = n / 2
for i in np.arange(0,int(n+1)):
    y = y + ((-1)**i * (x)**(2*i+1)) / factorial(2*i+1)
plt.plot(x, y, label = 'Eleventh Order', marker='.')

# Plotting of full solution, i.e. sin(x)
plt.plot(x, np.sin(x), 'k', label = '$\sin(x)$')
plt.grid()
plt.title('Taylor Series Approximations of $\sin(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()