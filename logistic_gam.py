import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from utils import (svd_whiten,
                   sigmoid,
                   B_spline,
                   safe_reciprocal)


class LogisticGAMGCV(BaseEstimator, ClassifierMixin):

        def __init__(self, ls=20., knots=10, maxiter=100):
                self.ls = ls
                self.knots = knots
                self.maxiter = maxiter


        def fit(self, X, Y):

                lambdas = np.logspace(-5, 2, self.ls)
                self.gcvs = np.array([
                              self._fit(X, Y, l)[1] for l in lambdas])

                best_l = lambdas[self.gcvs.argmin()]
                self.a, _ = self._fit(X, Y, best_l)


        def predict_proba(self, X):
                
                return sigmoid(self.a.dot(B_spline(X).T))
                

        def _gcv(self, B, D, y, y_, l):

                r = B.shape[0] * np.linalg.norm(y - y_)
                B = svd_whiten(B)
                trH = np.sum(np.einsum('ij, ij -> j',
                        np.linalg.pinv(B.T.dot(B) + l*D.dot(D.T)), B.T.dot(B)))

                trIA = (B.shape[1] - trH)**2

                return np.sum(r / trIA)


        def _fit(self, X, y, l):

                B = B_spline(X, n=self.knots)
                n, f = B.shape

                D = np.diff(np.eye(f), 2)

                a = np.zeros(f)
                p = sigmoid(a.dot(B.T))
                W = p * (1 - p)

                a = np.linalg.pinv((B.T * W).dot(B) +
                        l * D.dot(D.T)).dot(B.T).dot(y).reshape((f,))

                delta = 1
                i = 0
                eps = 1e-4
                while delta > eps and i < self.maxiter:

                        p  = sigmoid(a.dot(B.T))
                        W  = p * (1 - p)
                        W_ = safe_reciprocal((p * (1 - p)))
                        z  = B.dot(a) + np.multiply(W_.T, (y - p))

                        BTWBD = ((B.T * W).dot(B)
                                + l * D.dot(D.T))

                        a_n = np.linalg.pinv(BTWBD).dot(B.T * W).dot(z)

                        delta = np.linalg.norm(a_n - a)
                        a = a_n
                        i += 1

                y_ = sigmoid(a.dot(B.T))
                return a, self._gcv(B, D, y, y_, l)
