# dkesn
## install package



`pip install -r requirements.txt`

## run
```python
import matplotlib.pyplot as plt
import numpy as np

from kesn.core import KernelESN
from kesn.core import ESN

data = np.loadtxt(('./data.txt'))

np.random.seed(12345)

reg_l2 = ESN(reservoir_size=900, leak_rate=0.2603, rho=1.4857, init_len=1000, solver='l2', penalty=0.001, l1_ratio=0.4722)
reg_l1 = ESN(reservoir_size=700, leak_rate=0.2948, rho=0.8285, init_len=1000, solver='l1', penalty=0.0001, l1_ratio=0.4722)
reg_l05 = ESN(reservoir_size=700, leak_rate=0.2879, rho=0.9714, init_len=1000, solver='l0.5', penalty=1e-08, l1_ratio=0.4722)
reg_l1l2 = ESN(reservoir_size=800, leak_rate=0.2982, rho=1.1428, init_len=1000, solver='l1_l2', penalty=1e-07, l1_ratio=0.4722)
reg = KernelESN(reservoir_size=800, leak_rate=0.3000, rho=1.2500, init_len=1000, solver='krr', penalty=1e-8, l1_ratio=0.47222)

y_pred, y_true = reg.fit(data=data, train_size=2000, test_size=1000)
print('done.')
y_pred_l2, y_true_l2 = reg_l2.fit(data=data, train_size=2000, test_size=1000)
print('done.')
y_pred_l1, y_true_l1 = reg_l1.fit(data=data, train_size=2000, test_size=1000)
print('done.')
y_pred_l05, y_true_l05 = reg_l05.fit(data=data, train_size=2000, test_size=1000)
print('done.')
y_pred_l12, y_true_l12 = reg_l1l2.fit(data=data, train_size=2000, test_size=1000)
print('done.')

plt.subplot(5, 1, 1)
plt.plot(y_true[700:], 'r-', label='Raw data')
plt.plot(y_pred[700:], 'g-', label='DKESN')
plt.legend()

plt.subplot(5, 1, 2)
plt.plot(y_true_l2[700:], 'r-', label='Raw data')
plt.plot(y_pred_l2[700:], 'g-', label='RESN')
plt.legend()

plt.subplot(5, 1, 3)
plt.plot(y_true_l1[700:], 'r-', label='Raw data')
plt.plot(y_pred_l1[700:], 'g-', label='LESN')
plt.legend()

plt.subplot(5, 1, 4)
plt.plot(y_true_l12[700:], 'r-', label='Raw data')
plt.plot(y_pred_l12[700:], 'g-', label='EESN')
plt.legend()

plt.subplot(5, 1, 5)
plt.plot(y_true_l05[700:], 'r-', label='Raw data')
plt.plot(y_pred_l05[700:], 'g-', label='$\ell_{1/2}$-ESN')
plt.legend()

plt.show()

pred300 = np.asarray([y_pred, y_pred_l2, y_pred_l1, y_pred_l12, y_pred_l05])
np.savetxt('pred300.txt', pred300, fmt='%.8f')


