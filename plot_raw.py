import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

data = np.loadtxt('data.txt')


n = 500
x = [i for i in range(n)]

plt.plot(x, data[:n], linewidth=1.6)
plt.xlabel('$t$')
plt.ylabel('$x(t)$')
plt.show()
