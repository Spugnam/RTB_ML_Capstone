from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
# import mmh3 # commenting to run with pypy
# import pandas as pd
# from sklearn.feature_selection import mutual_info_classif

# parameters #################################################################

train = '../data/mini_train.csv'  # path to training file
test = '../data/mini_test.csv'  # path to testing file

# create test panda df
# test_df = pd.read_csv(test)
# test_df = test_df.drop(test_df.columns[0], axis=1)

D = 2 ** 20   # number of weights use for learning
alpha = .1    # learning rate for stochastic gradient descent (sgd) optimization


# function definitions #######################################################

# A. Bounded logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)


# B. Apply hash trick of the original csv row
# for simplicity, we treat both integer and categorical features as categorical
# INPUT:
#     csv_row: a csv dictionary, ex: {'Lable': '1', 'I1': '357', 'I2': '', ...}
#     D: the max index that we can hash to
# OUTPUT:
#     x: a list of indices that its value is 1
def get_x(csv_row, D):
    x = [0]  # 0 is the index of the bias term
    for key, value in csv_row.items():
        index = hash(value + key[1:]) % D
        x.append(index)
    return x  # x contains indices of features that have a value of 1


# C. Get probability estimation on x
# INPUT:
#     x: features
#     w: weights
# OUTPUT:
#     probability of p(y = 1 | x; w)
def get_p(x, w):
    wTx = 0.
    for i in x:  # do wTx
        wTx += w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid


# D. Update given model
# INPUT:
#     w: weights
#     n: a counter that counts the number of times we encounter a feature
#        this is used for adaptive learning rate
#     x: feature
#     p: prediction of our model
#     y: answer
# OUTPUT:
#     w: updated model
#     n: updated count
def update_w(w, n, x, p, y):
    for i in x:
        # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
        # (p - y) * x[i] is the current gradient
        # note that in our case, if i in x then x[i] = 1
        w[i] -= (p - y) * alpha / (sqrt(n[i]) + 1.)  # gradient descent formula with adaptive alpha
        n[i] += 1.

    return w, n


# training and testing #######################################################

# initialize our model
w = [0.] * D  # weights
n = [0.] * D  # number of times we've encountered a feature

# start training a logistic regression model using on pass sgd
loss = 0.
for t, row in enumerate(DictReader(open(train))):
    y = 1. if row['clicked'] == '1' else 0.

    del row['clicked']  # can't let the model peek the answer
    # del row['day']  # only used to split train/ test, timestamp_weekday for learning

    # main training procedure
    # step 1, get the hashed features
    x = get_x(row, D)

    # step 2, get prediction
    p = get_p(x, w)

    # for progress validation, useless for learning our model
    loss += logloss(p, y)
    if t % 1e6 == 0 and t > 1:
        print('%s\tencountered: %d\tcurrent logloss: %f' % (
            datetime.now(), t, loss / t))

    # step 3, update model with answer
    w, n = update_w(w, n, x, p, y)

# save probability column
# test_df['probability'] = 0
for t, row in enumerate(DictReader(open(test))):
    x = get_x(row, D)
    p = get_p(x, w)
    # test_df.loc[t, 'probability'] = p

# Saving to file
with open(str(datetime.now()).split(".")[0] + 'predictions.csv', 'w') as predictions:
    predictions.write('Id,Predicted\n')
    for t, row in enumerate(DictReader(open(test))):
        Id = t
        x = get_x(row, D)
        p = get_p(x, w)
        _ = predictions.write('%s,%f\n' % (Id, p))

Mutual information ############################################

MI = mutual_info_classif(test_df.select_dtypes(include=['int64']),
                         test_df['clicked'].values)
MI_list = list()
for i in range(len(MI)):
    MI_list.append([test_df.select_dtypes(include=['int64']).columns[i], MI[i]])

MI_list.sort(key=lambda x: -x[1])
print(MI_list, "\n")
# Results #######################################################

# logloss
from sklearn.metrics import log_loss
print("log_loss: ",
      log_loss(test_df['clicked'], test_df['probability'], eps=1e-15, normalize=True),
      "\n")
# Normalize = True -> return the mean loss per sample

from sklearn.metrics import roc_auc_score
print("roc_auc_score",
      roc_auc_score(test_df['clicked'], test_df['probability'], average="macro"),
      "\n")
# roc_auc_score(['clicked'], test_df['probability'], average="weighted")

from sklearn.metrics import average_precision_score
print("average_precision_score",
      average_precision_score(test_df['clicked'], test_df['probability'], average="macro"))

test_df.loc[:, ['clicked', 'probability']].head(20)
