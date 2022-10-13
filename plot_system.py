import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st
from kesn.utils import get_interval

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

data = np.loadtxt('data.txt')

# plt.plot([i for i in range(1, 501)], data[:500], 'b-')
# plt.xlabel(r'$\tau$')
# plt.ylabel(r'$x\left( \tau \right)$')
# plt.show()

loss_1 = np.loadtxt('ens-l2.txt')
loss_2 = np.loadtxt('ens-l1.txt')
loss_3 = np.loadtxt('ens-l0.5.txt')
loss_4 = np.loadtxt('ens-l1_l2.txt')
loss_5 = np.loadtxt('ens-krr.txt')

m1, int1 = get_interval(loss_1)
m2, int2 = get_interval(loss_2)
m3, int3 = get_interval(loss_3)
m4, int4 = get_interval(loss_4)
m5, int5 = get_interval(loss_5)

step = [i for i in range(6)]

plt.plot(m1, 'r-o', label='RESN')
plt.fill_between(x=step, y1=int1[0], y2=int1[1], alpha=0.3, color='r')

plt.plot(m2, 'g-^', label='LESN')
plt.fill_between(x=step, y1=int2[0], y2=int2[1], alpha=0.3, color='g')

plt.plot(m3, 'b-v', label='$\ell_{1/2}$-ESN')
plt.fill_between(x=step, y1=int3[0], y2=int3[1], alpha=0.3, color='b')

plt.plot(m4, 'y-d', label='EESN')
plt.fill_between(x=step, y1=int4[0], y2=int4[1], alpha=0.3, color='y')

plt.plot(m5, 'k-s', label='DKESN')
plt.fill_between(x=step, y1=int5[0], y2=int5[1], alpha=0.3, color='k')

xticklabes = [500, 1000, 1500, 2000, 2500, 3000]
plt.xticks(step, xticklabes)
plt.xlabel('Number of test data')
plt.ylabel('MSE')

plt.legend()

plt.show()
