import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from logistic_gam import LogisticGAMGCV


if __name__ == '__main__':

        x = np.linspace(0,10,200).ravel()
        y = x**2 - 10*x + np.random.normal(0, 1, 200)
        y = (y > y.mean()).astype(int)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                test_size=0.33, random_state=42)

        clf = LogisticGAMGCV()
        clf.fit(x_train, y_train)
        p = clf.predict_proba(x_test)


        print('Log Reg:', roc_auc_score(y_test, LogisticRegression().fit(
                x_train.reshape(-1,1), y_train).predict_proba(
                x_test.reshape(-1,1))[:, 1]))

        print('This Model:', roc_auc_score(y_test, p))

