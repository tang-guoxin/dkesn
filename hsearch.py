import hyperopt
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def fun(p):
    x, y = p['x'], p['y']
    loss = x + y
    return {'loss': loss, 'status': STATUS_OK}


space = {
    'x': hp.choice('x', [1, -1, 3]),
    'y': hp.choice('y', [4, -2, 6])
}

trails = Trials()

best = fmin(fn=fun, space=space, algo=tpe.suggest, max_evals=20, trials=trails)

print(best)


ls = trails.losses()

plt.plot(ls)
plt.show()

num = hp.randint('a', 1, 10)
