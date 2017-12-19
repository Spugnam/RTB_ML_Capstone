from datetime import datetime
from math import exp, log, sqrt
import mmh3     # import murmurhash for feature hashing
from pyspark.sql import Row
import sys
reload(sys)
sys.setdefaultencoding('utf8') # prevents ASCII crash

def update(self, x, p, y):
    ''' Update model using x, p, y
    INPUT:
    x: feature, a list of indices
    p: click probability prediction of our model
    y: answer

    MODIFIES:
    self.n: increase by squared gradient
    self.z: weights
    '''
    # parameter
    alpha = self.alpha

    # model
    n = self.n
    z = self.z
    w = self.w

    # gradient under logloss
    g = p - y

    # update z and n
    for i in self._indices(x):
        sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
        z[i] += g - sigma * w[i]
        n[i] += g * g

def get_prob(self, x):
    ''' Get probability estimation on x

        INPUT:
            x: features

        OUTPUT:
            probability of p(y = 1 | x; w)
    '''

    # parameters
    alpha = self.alpha
    beta = self.beta
    L1 = self.L1
    L2 = self.L2

    # model
    n = self.n
    z = self.z
    w = {}

    # wTx is the inner product of w and x
    wTx = 0.
    for i in self._indices(x):
        sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

        # build w on the fly using z and n, hence the name - lazy weights
        # we are doing this at prediction instead of update time is because
        # this allows us for not storing the complete w
        if sign * z[i] <= L1:
            # w[i] vanishes due to L1 regularization
            w[i] = 0.
        else:
            # apply prediction time L1, L2 regularization to z and get w
            w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

        wTx += w[i]

    # cache the current w for update stage
    self.w = w

    # bounded sigmoid function, this is the probability estimation
    return 1. / (1. + exp(-__builtins__.max(__builtins__.min(wTx, 35.), -35.)))
    # prevent overwriting with pyspark.sql.functions min and max

def data(df, D):
''' GENERATOR: Apply hash-trick to input dataframe
               and for simplicity, we one-hot-encode everything

    INPUT:
        df: dataframe to train on
        D: the max index that we can hash to

    YIELDS:
        x: a list of hashed and one-hot-encoded 'indices'
           we only need the index since all values are either 0 or 1
        y: y = 1 if we have a click, else we have y = 0
'''
    for t, row in enumerate(df.rdd.toLocalIterator()):
        # process clicks
        y = 0.
        if 'clicked' in row:
          if row['clicked'] == 1:
            y = 1.

        # build x
        x = []
        for key in df.columns:
          if (key != 'clicked'): # do not use label!
            value = str(row[key]).encode('utf-8')
            # one-hot encode everything with hash trick
            index = mmh3.hash(key + '_' + value, signed=False) % D
            x.append(index)

        yield t, x, y


def fit(self, df):
    ''' Fit model
    '''

    D = self.D
    epoch = self.epoch

    # start training
    for e in xrange(epoch):

      for t, x, y in data(df, D):  # get hashed features
        #    t: just a instance counter
        #    x: features
        #    y: label (click)

        # get prediction from learner
        p = self.get_prob(x)

        # update weights
        self.update(x, p, y)

def predict(self, df):

    D = self.D
    prob_list = list()
    clicks_list = list()

    for t, x, y in data(df, D):
      p = self.get_prob(x)

      prob_list.append(p)
      clicks_list.append(y)

    return prob_list, clicks_list
