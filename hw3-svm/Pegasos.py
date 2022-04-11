from load import folder_list
import pickle
import random
from collections import Counter
from myutil import *


def shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = "data/pos"
    neg_path = "data/neg"

    pos_review = folder_list(pos_path, 1)
    neg_review = folder_list(neg_path, -1)

    review = pos_review + neg_review
    random.shuffle(review)
    pickle.dump(review, open("save.p", "wb"))


'''
Now you have read all the files into list 'review' and it has been shuffled.
Save your shuffled result by pickle.
*Pickle is a useful module to serialize a python object structure. 
*Check it out. https://wiki.python.org/moin/UsingPickle
'''


def split_train_valid():
    review = pickle.load(open("save.p", "rb"))
    train = review[0:1500]
    valid = review[1500:2000]
    return train, valid


def text2bag(text):
    cnt = Counter()
    for word in text:
        cnt[word] += 1
    return cnt


def cal_ER(train, w, lamb):
    cost = 0
    for text in train:
        label = int(text[-1])
        featrue = text2bag(text)
        cost += max(0, 1 - label * dotProduct(w, featrue))
    cost /= len(train)
    cost += lamb / 2 * dotProduct(w, w)
    return cost


def Pegasos_slow(train, lamb, epochs):
    w = Counter()
    t = 0
    for epoch in range(epochs):
        for text in train:  # round robin stochastic sub-gradient descent
            label = int(text[-1])
            feature = text2bag(text)
            t += 1
            step_size = 1 / (t * lamb)
            increment(w, - step_size * lamb, w)
            if dotProduct(w, feature) * label < 1:
                increment(w, step_size * label, feature)
    return w


def Pegasos_fast(train, lamb):
    W = Counter()
    s = 1
    t = 0
    while cal_ER(train, W, lamb) >= 1:
        increment(W, 1 / s - 1, W)
        for text in train:  # round robin stochastic sub-gradient descent
            label = int(text[-1])
            feature = text2bag(text)
            t += 1
            step_size = 1 / (t * lamb)
            if s * dotProduct(W, feature) * label < 1:
                s = (1 - step_size * lamb) * s
                if s == 0:
                    s = 1
                    W = Counter()
                increment(W, (1 / s) * step_size * label, feature)
            else:
                s = (1 - step_size * lamb) * s
                if s == 0:
                    s = 1
                    W = Counter()
        increment(W, s - 1, W)
    return W


def Pegasos_extreme_fast(train, lamb):
    theta = Counter()  # theta_1 = 0
    t = 1
    while cal_ER(train, theta,lamb) >= 0.1:
        increment(theta, -lamb*(t-1)-1, theta) # we restore the origin value of theta here for following normal eval
        for text in train:
            label = int(text[-1])
            feature = text2bag(text)
            if t == 1 or -1 / (lamb * (t - 1)) * dotProduct(theta, feature) * label < 1:
                increment(theta, -label, feature)
            t += 1
        increment(theta, -1/(lamb * (t-1))-1, theta)  # we store w_t in theta for calculating empirical risk
    return theta


def verify_pegasos(w_fast, w_slow):
    for f, v in w_fast.items():
        if abs(w_slow.get(f, 0) - v) > 1e-4:
            print("verified failed!\n")
            print("w_fast[" + f + "] = " + v + "\n")
            print("w_slow[" + f + "] = " + w_slow.get(f, 0) + "\n")
            return False
    print("verified success!\n")
    return True


def eval_error_rate(w, valid):
    error_num = 0
    for text in valid:
        label = int(text[-1])
        feature = text2bag(text)
        if dotProduct(w, feature) * label <= 0:  # mis-classified data
            error_num += 1
    print("error rate is: " + str((float(error_num)) / len(valid)))


if __name__ == '__main__':
    # shuffle_data()
    train, valid = split_train_valid()
    # lamb = 0.0008
    lamb = 0.01
    # w_fast = Pegasos_fast(train, lamb)
    w_extreme_fast = Pegasos_extreme_fast(train, lamb)
    # eval_error_rate(w_fast, valid)
    eval_error_rate(w_extreme_fast, valid)
    # print(w_extreme_fast)
