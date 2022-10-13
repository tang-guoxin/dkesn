from matplotlib import pyplot as plt
from kesn.core import ESN
from kesn.core import KernelESN
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

data = np.loadtxt('data.txt')

r_ls = [i*100 for i in range(1, 11)]
leak_ls = np.linspace(0.25, 0.35, 30).tolist()
rho_ls = np.linspace(0.8, 1.5, 50).tolist()
p_ls = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
beta_ls = np.linspace(0.25, 0.75, 10).tolist()
mu_ls = np.linspace(0.25, 0.75, 10).tolist()

space_esn = {
    'reservoir_size': hp.choice('reservoir_size', r_ls),
    'leak_rate': hp.choice('leak_rate', leak_ls),
    'rho': hp.choice('rho', rho_ls),
    'penalty': hp.choice('penalty', p_ls)
}

# space_esn = {
#     'reservoir_size': hp.choice('reservoir_size', r_ls),
#     'leak_rate': hp.choice('leak_rate', leak_ls),
#     'rho': hp.choice('rho', rho_ls),
#     'penalty': hp.choice('penalty', p_ls),
#     'mu': hp.choice('mu', mu_ls)
# }

# space_esn = {
#     'reservoir_size': hp.choice('reservoir_size', r_ls),
#     'leak_rate': hp.choice('leak_rate', leak_ls),
#     'rho': hp.choice('rho', rho_ls),
#     'penalty': hp.choice('penalty', p_ls),
#     'beta': hp.choice('beta', beta_ls)
# }


def esn_loss(input_dict):
    reservoir_size = input_dict['reservoir_size']
    leak_rate = input_dict['leak_rate']
    rho = input_dict['rho']
    penalty = input_dict['penalty']
    # mu = input_dict['mu']
    # beta = input_dict['beta']
    # mu = 0.5
    # beta = 0.5
    # reg = ESN(reservoir_size=reservoir_size, leak_rate=leak_rate, rho=rho, init_len=1000, solver='quantile', penalty=penalty, l1_ratio=mu, q_score=beta)
    reg = KernelESN(reservoir_size=reservoir_size, leak_rate=leak_rate, rho=rho, init_len=1000, solver='krr', penalty=penalty)
    reg.fit(data, train_size=3000, test_size=2000)
    loss = reg.score()
    if loss > 0.1:
        loss = 0.1
    return {'loss': loss, 'status': STATUS_OK}


trails = Trials()
best = fmin(fn=esn_loss, space=space_esn, algo=tpe.suggest, max_evals=1000, trials=trails)
print(best)
best_pars = {'reservoir_size': r_ls[best.get('reservoir_size')],
             'leak_rate': leak_ls[best.get('leak_rate')],
             'rho': rho_ls[best.get('rho')],
             'penalty': p_ls[best.get('penalty')],
             # 'beta': mu_ls[best.get('beta')]
             }

print(best_pars)

ls = trails.losses()
plt.plot(ls)
plt.show()
