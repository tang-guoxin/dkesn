import matplotlib.pyplot as plt
import numpy as np
from celer import Lasso as WeightLasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import QuantileRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoLars
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from sklearn.metrics.pairwise import rbf_kernel
from scipy import sparse
from statsmodels.formula.api import quantreg
from statsmodels.regression.quantile_regression import QuantReg
import warnings


warnings.filterwarnings('ignore')

try:
    from .utils import rho_w
    from .utils import calc_sigam
except:
    from utils import rho_w
    from utils import calc_sigam


class KernelESN:
    def __init__(self, reservoir_size=999, leak_rate=0.3, rho=1.25, penalty=1e-8, sigma=None, max_iter=100,
                 init_len=100, eps=1e-24, verbose=False, input_dim=1, solver='l2', density=0.2, l1_ratio=0.5, q_score=0.5):
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        self.rho = rho
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.sigma = sigma
        self.max_iter = max_iter
        self.init_len = init_len
        self.centers = None
        self.eps = eps
        self.verbose = verbose
        self.y_pred = None
        self.y_test = None
        self.input_dim = input_dim
        self.train_data = None
        self.solver = solver
        self.density = density
        self.q_score = q_score

    def isprint(self, info):
        if self.verbose:
            print(info)

    def scale_wr(self):
        # wr = np.random.random((self.reservoir_size, self.reservoir_size)) - 0.5
        sw = sparse.rand(m=self.reservoir_size, n=self.reservoir_size, density=self.density)
        wr = sw.toarray() - 0.5
        wr[wr == -0.5] = 0
        wr = wr * 2
        p = rho_w(wr)
        res = (self.rho / p) * wr
        return res

    def weight_lasso(self, state, yt):
        loss = list()
        # [_, n_feature] = state.shape
        # weights = np.ones(n_feature)
        reg_l = LassoLars(alpha=self.penalty, fit_intercept=False)
        reg_l.fit(state, yt)
        beta = reg_l.coef_
        weights = 1 / (np.sqrt(np.abs(beta)) + self.eps)
        beta_ = beta
        for i in range(self.max_iter):
            reg = WeightLasso(alpha=self.penalty, weights=weights, fit_intercept=False, max_epochs=200, max_iter=200)
            reg.fit(state, yt)
            loss.append(reg.score(state, yt))
            beta = reg.coef_
            weights = 1 / (np.sqrt(np.abs(beta)) + self.eps)
            dert = np.linalg.norm(beta_ - beta)
            if self.verbose:
                print(f'\r solve l_1/2: step = {i + 1} / {self.max_iter}, eps = {dert}.', end='')
            if dert < 1e-2:
                break
            beta_ = beta
        self.isprint('\n')
        return beta, loss

    def solve(self, state, yt, lambda_1=1e-8, solver='l2'):
        reg = Ridge(alpha=lambda_1, solver='svd', fit_intercept=False)
        # w = np.linalg.pinv(state @ state.T + lambda_1*np.eye(self.reservoir_size+2)) @ state @ np.reshape(yt, (-1, 1))
        # w_o = w[:, 0]
        if solver == 'l2':
            self.isprint('solver = Ridge.')
        if solver == 'l1':
            self.isprint('solver = Lasso.')
            reg = Lasso(alpha=lambda_1, fit_intercept=False)
        if solver == 'l1_l2':
            self.isprint('solver = ElasticNet.')
            reg = ElasticNet(alpha=self.penalty, l1_ratio=self.l1_ratio, fit_intercept=False)
        if solver == 'huber':
            self.isprint('solver = Huber.')
            reg = HuberRegressor(alpha=lambda_1, fit_intercept=False)
        if solver == 'bayes':
            self.isprint('solver = Huber.')
            reg = BayesianRidge(fit_intercept=False)
        if solver == 'quantile':
            self.isprint('solver = Quantile.')
            reg = QuantileRegressor(quantile=self.q_score, alpha=self.penalty, fit_intercept=False, solver='highs-ds')
            # reg = QuantReg(endog=yt, exog=state.T)
            # reg.fit(q=self.q_score)
            # return reg
        if solver == 'krr':
            self.isprint('solver = krr.')
            reg = KernelRidge(alpha=self.penalty, kernel='rbf')
            reg.fit(state.T, yt)
            return reg
        if solver == 'l0.5':
            self.isprint('solver = l_1/2.')
            beta, loss = self.weight_lasso(state.T, yt)
            return beta
        # print(reg.__str__())
        # reg = QuantileRegressor(quantile=self.q_score, alpha=self.penalty, fit_intercept=False, solver='highs-ds')
        reg.fit(state.T, yt)
        w_out = reg.coef_
        return w_out

    def fit(self, data=None, train_size=3000, test_size=500):
        inputs = np.vstack((np.ones_like(data), data)).T
        self.train_data = inputs
        [_, n] = np.shape(inputs)
        clu = KMeans(n_clusters=self.reservoir_size)
        clu.fit(inputs[:train_size, :])
        self.centers = clu.cluster_centers_
        if self.sigma is None:
            self.sigma = calc_sigam(x=inputs[:train_size])
            self.isprint(f'sigma = {self.sigma}.')
        wr = self.scale_wr()
        state = np.zeros((n + self.reservoir_size, train_size - self.init_len))
        yt = data[self.init_len + 1:train_size + 1]
        x = np.zeros((self.reservoir_size, 1))
        self.isprint('training...')
        kermat = rbf_kernel(inputs[:train_size+2, :], self.centers, gamma=1/self.sigma)
        for t in range(train_size):
            if self.verbose:
                print(f'\r train step = {round(100 * (t + 1) / train_size, 4)}%.', end='')
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(
                np.reshape(kermat[t, :], (-1, 1)) + wr @ x)
            if t >= self.init_len:
                state[:, t - self.init_len] = np.hstack((inputs[t, :], x[:, 0]))
        self.isprint('\n train done.')
        self.isprint('solve...')
        w_out = self.solve(state, yt, solver=self.solver)
        self.isprint('solve done.')
        y_pred = np.zeros(test_size)
        # ! predict...
        self.isprint('predict...')
        u = inputs[train_size, :]
        for t in range(test_size):
            if self.verbose:
                print(f'\r predict step = {round(100 * (t + 1) / test_size, 4)}%.', end='')
            km_ = rbf_kernel(np.reshape(u, (1, -1)), self.centers, gamma=1/self.sigma)
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(np.reshape(km_, (-1, 1)) + wr @ x)
            s = np.vstack((np.reshape(u, (-1, 1)), x))
            if self.solver == 'krr':
                y_hat = w_out.predict(s.T)
                u = np.asarray([1, y_hat])
                y_pred[t] = y_hat
            else:
                y_hat = w_out @ s
                y_hat = y_hat[0]
                u = np.asarray([1, y_hat])
                y_pred[t] = y_hat
            # u = np.asarray([1, data[train_size+t+1]])
        self.isprint('\n predict done.')
        y_true = data[train_size + 1:train_size + test_size + 1]
        self.y_test = y_true
        self.y_pred = y_pred
        return y_pred, y_true

    def score(self):
        # pars = self.reservoir_size + self.input_dim + 1
        # n_sample = self.train_data.shape[0]
        try:
            mse = mean_squared_error(y_true=self.y_test, y_pred=self.y_pred)
        except:
            mse = np.inf
        # rss = np.sum(np.power(self.y_test - self.y_pred, 2))
        # bic = pars * np.log(n_sample) - np.log(rss / n_sample)
        return mse


class ESN(KernelESN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, data=None, train_size=3000, test_size=500):
        inputs = np.vstack((np.ones_like(data), data)).T
        self.train_data = inputs
        [_, n] = np.shape(inputs)
        wr = self.scale_wr()
        state = np.zeros((n + self.reservoir_size, train_size - self.init_len))
        yt = data[self.init_len + 1:train_size + 1]
        x = np.zeros((self.reservoir_size, 1))
        self.isprint('training...')
        w_in = np.random.random((self.reservoir_size, n))
        for t in range(train_size):
            if self.verbose:
                print(f'\r train step = {round(100 * (t + 1) / train_size, 4)}%.', end='')
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(w_in @ np.reshape(inputs[t, :], (-1, 1)) + wr @ x)
            if t >= self.init_len:
                state[:, t - self.init_len] = np.hstack((inputs[t, :], x[:, 0]))
        self.isprint('\n train done.')
        self.isprint('solve...')
        w_out = self.solve(state, yt, solver=self.solver)
        self.isprint('solve done.')
        y_pred = np.zeros(test_size)
        # ! predict...
        self.isprint('predict...')
        u = inputs[train_size, :]
        for t in range(test_size):
            if self.verbose:
                print(f'\r predict step = {round(100 * (t + 1) / test_size, 4)}%.', end='')
            x = (1 - self.leak_rate) * x + self.leak_rate * np.tanh(w_in @ np.reshape(u, (-1, 1)) + wr @ x)
            s = np.vstack((np.reshape(u, (-1, 1)), x))
            if self.solver == 'krr':
                y_hat = w_out.predict(s.T)
                u = np.asarray([1, y_hat])
                y_pred[t] = y_hat
            else:
                y_hat = w_out @ s
                y_hat = y_hat[0]
                u = np.asarray([1, y_hat])
                y_pred[t] = y_hat
            # u = np.asarray([1, data[train_size+t+1]])
        self.isprint('\n predict done.')
        y_true = data[train_size + 1:train_size + test_size + 1]
        self.y_test = y_true
        self.y_pred = y_pred
        return y_pred, y_true


if __name__ == '__main__':
    data = np.load('../data.npy')
    data = data[1999:, 1]
    # data = np.loadtxt('../data.txt')
    # 0.028963773003522132,
    # sigma=0.010821681721668152,
    print(data.shape)

    KESN = KernelESN(reservoir_size=1000,
                     init_len=1000,
                     leak_rate=0.3,
                     penalty=1e-8,
                     rho=1.25,
                     max_iter=30,
                     sigma=None,
                     solver='krr',
                     verbose=True)

    y1, y2 = KESN.fit(data, train_size=3000, test_size=1000)

    print(KESN.score())

    plt.plot(y1, 'g-')
    plt.plot(y2, 'r-')
    plt.show()


