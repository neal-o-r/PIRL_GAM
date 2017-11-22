import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from spline_utils import (svd_whiten,
                          sigmoid,
                          B_spline,
                          div0)


def fit(X, y, maxiter=100, eps=1e-4, l=1):

        B = B_spline(X)
        n, f = B.shape

        D = np.diff(np.eye(f), 2)

        a = np.zeros(f)
        p = sigmoid(a.dot(B.T))
        W = p * (1 - p)

        a = np.linalg.pinv((B.T * W).dot(B) +
                        l * D.dot(D.T)).dot(B.T).dot(y).reshape((f,))

        delta = 1

        i = 0
        while (delta > eps) and i < maxiter:

                p  = sigmoid(a.dot(B.T))
                W  = p * (1 - p)
                W_ = div0((p * (1 - p)))
                z  = B.dot(a) + np.multiply(W_.T, (y - p))

                BTWBD = ((B.T * W).dot(B) 
                                + l * D.dot(D.T))

                a_n = np.linalg.pinv(BTWBD).dot(B.T * W).dot(z)

                delta = np.linalg.norm(a_n - a)
                a = a_n
                i += 1

        y_ = sigmoid(a.dot(B.T))
        return a, gcv_score(B, W, D, y, y_, l)


def gcv_score(B, W, D, y, y_, l):

        r = B.shape[0] * np.linalg.norm(y - y_)
        B = svd_whiten(B)
        trH = np.sum(np.einsum('ij, ij -> j',
                        np.linalg.pinv(B.T.dot(B) + l*D.dot(D.T)), B.T.dot(B)))

        trIA = (B.shape[1] - trH)**2

        return np.sum(r / trIA)


if __name__ == '__main__':

        x = np.linspace(0,10,200).ravel()
        y = x**2 - 10*x + np.random.normal(0, 1, 200)
        y = (y > y.mean()).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                test_size=0.33, random_state=42)

        ls = np.logspace(-5, 2, 20)
        gs = np.array([fit(x_train, y_train, l=i)[1] for i in ls])

        a, g = fit(x_train, y_train, l=ls[gs.argmin()])
        p = sigmoid(a.dot(B_spline(x_test).T))

        print('Log Reg:', roc_auc_score(y_test, LogisticRegression().fit(
                x_train.reshape(-1,1), y_train).predict_proba(
                x_test.reshape(-1,1))[:, 1]))

        print('This Model:', roc_auc_score(y_test, p))

        plt.plot(ls, gs, '-o')
        plt.loglog()
        plt.show()
