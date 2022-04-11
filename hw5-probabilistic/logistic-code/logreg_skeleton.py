import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn import preprocessing


class my_LogisticRegression:
    def __init__(self,X_train, y_train, X_val, y_val, l2reg=1.0):
        if l2reg < 0:
            raise ValueError('Regularization penalty should be at least 0.')
        self.l2reg = l2reg
        self.X_train =X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val =y_val

    def fit(self):
        n, num_ftrs = self.X_train.shape

        def logistic_obj(w):
            res = 0
            for i in range(n):
                res += np.logaddexp(0, -self.y_train[i] * np.dot(w, self.X_train[i]))
            res /= n
            res += self.l2reg * np.dot(w, w)
            return res

        w_0 = np.zeros(num_ftrs)
        self.w = minimize(logistic_obj, w_0).x
        return self

    def negative_log_likelihood(self):
        try:
            getattr(self, "w")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        n = len(self.y_val)
        nll = 0
        for i in range(n):
            nll += np.logaddexp(0, -self.y_val[i] * np.dot(self.w, self.X_val[i]))
        return nll

    def pred_prob(self):
        self.fit()
        prob_arr = []
        for i in range(self.X_val.shape[0]):
            prob_arr.append(1/(1+np.exp(-np.dot(self.w, self.X_val[i]))))
        return prob_arr


def load_problem(X_file_name, y_file_name):
    X = np.loadtxt(X_file_name, delimiter=',')
    y = np.loadtxt(y_file_name)
    return X, y

def preprocessing_data():
    X_train, y_train = load_problem("X_train.txt", "y_train.txt")
    X_val, y_val = load_problem("X_val.txt", "y_val.txt")
    # pre-processing training set
    scalar = preprocessing.StandardScaler()
    X_train = scalar.fit_transform(X_train)  # standardizing the data
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # adding a column for the bias term.
    y_train[y_train == 0] = -1
    # pre-processing validation set
    scalar = preprocessing.StandardScaler()
    X_val = scalar.fit_transform(X_val)
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))
    y_val[y_val == 0] = -1
    return X_train, y_train, X_val, y_val

def plot_likelihood_l2reg(X_train, y_train, X_val, y_val):
    power = np.arange(-5, 3, 0.3)
    l2_reg = [10**i for i in power]
    w_arr = []
    log_lh = []
    for l2 in l2_reg:
        my_clf = my_LogisticRegression(X_train,y_train,X_val,y_val,l2)
        w_arr.append(my_clf.fit())
        log_lh.append(my_clf.negative_log_likelihood())
    fig, ax = plt.subplots()
    ax.plot(l2_reg, log_lh)
    ax.set_xscale('log')
    ax.set_xlabel('l2reg')
    ax.set_ylabel('log-likelihood')
    ax.set_title('log-likelihood v.s. l2reg')
    plt.show()

def plot_calibration(X_train, y_train, X_val, y_val):
    l2 = 0.02  # the optimal regularize we have found
    my_clf = my_LogisticRegression(X_train, y_train, X_val, y_val,l2)
    tol = 0.1
    f_x = np.linspace(0, 1, 10)
    prob_arr = my_clf.pred_prob()
    fractions = []
    # we create 10 bins for validation set
    # the i'th bin contains all data point that predict y to be 1 with probability
    # [(i-1)/10,(i+1)/10], and we compare it with fraction of y=1 in this bin
    for f in f_x:
        ind = np.where(np.abs(prob_arr-f) < tol)
        y_true = y_val[ind[0]]
        if(len(ind[0])==0):
            fraction = f
        else:
            fraction = np.sum([i == 1 for i in y_true])/len(ind[0])
        fractions.append(fraction)
    fig, ax =plt.subplots()
    ax.plot(f_x, fractions, '--', label='validation set', color='b')
    ax.plot(f_x, f_x, '--',label='perfectly calibrated', color='r')
    ax.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    X_train, y_train, X_val, y_val = preprocessing_data()
    # plot_likelihood_l2reg(X_train, y_train, X_val, y_val)
    plot_calibration(X_train, y_train, X_val, y_val)