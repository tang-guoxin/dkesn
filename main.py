from matplotlib import pyplot as plt
from kesn.core import ESN
from kesn.core import KernelESN
import numpy as np


data = np.loadtxt('data.txt')

best_pars = {
    'l2': {
        'reservoir_size': 900, 'leak_rate': 0.2603448275862069, 'rho': 1.4857142857142858, 'penalty': 0.001
    },
    'l1': {
        'reservoir_size': 700, 'leak_rate': 0.29482758620689653, 'rho': 0.8285714285714286, 'penalty': 0.0001
    },
    'l0.5': {
        'reservoir_size': 700, 'leak_rate': 0.2879310344827586, 'rho': 0.9714285714285715, 'penalty': 1e-08
    },
    'l1_l2': {
        'reservoir_size': 800, 'leak_rate': 0.2982758620689655, 'rho': 1.1428571428571428, 'penalty': 1e-07, 'mu': 0.4722222222222222
    },
    'krr': {
        'reservoir_size': 800, 'leak_rate': 0.3, 'rho': 1.25, 'penalty': 1e-8
    }
}

solver = 'l2'

pars = best_pars[solver]
reservoir_size = pars['reservoir_size']
leak_rate = pars['leak_rate']
rho = pars['rho']
penalty = pars['penalty']


loss_array = np.zeros((6, 20))
for row, test_size in enumerate([500, 1000, 1500, 2000, 2500, 3000]):
    loss_list = []
    for i in range(20):
        reg = ESN(reservoir_size=reservoir_size, leak_rate=leak_rate, rho=rho, init_len=1000, solver=solver, penalty=penalty, l1_ratio=0.4722)
        # reg = KernelESN(reservoir_size=reservoir_size, leak_rate=leak_rate, rho=rho, init_len=1000, solver='krr')
        y_pred, y_true = reg.fit(data, train_size=3000, test_size=test_size)
        loss = reg.score()
        loss_list.append(loss)
        print(f'i = {i+1}, test_size = {test_size}, loss = {loss}')
    loss_array[row, :] = np.asarray(loss_list)


np.savetxt('ens-' + solver + '.txt', loss_array, fmt='%.10f')
