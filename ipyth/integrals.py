import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as spi

x0 = -0.5
x1 = 1.5
n = 100

x = np.linspace(x0, x1, n)
y = np.exp(-x**2)

f = lambda x : np.exp(-x**2)

m = np.array([4,8,16])

fig, ax = plt.subplots(m.size,1)
for j,k in enumerate(m):
    nsteps = np.float(k)
    dx = 1.0 / nsteps
    x_approx = np.linspace(0.0, 1.0, num=k)
    T = spi.trapz(f(x_approx), x_approx)
    for i in np.arange(0, nsteps):
        x_start = i*dx
        x_stop = (i+1)*dx 
        y_start = np.exp(-x_start**2) 
        y_stop = np.exp(-x_stop**2)
        ax[j].fill_between([x_start,x_stop], [y_start,y_stop], facecolor='b', edgecolor='b', alpha=0.2)
        ax[j].scatter([x_start,x_stop], [y_start,y_stop])

    ax[j].plot(x, y, 'b-')
    ax[j].set_title('Trapezoid Rule, N = {}, Approx: {}'.format(k,T))
    ax[j].grid(True)
    ax[j].set_xlabel('x')
    ax[j].set_ylabel('y')

plt.show()
