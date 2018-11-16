"""calculate utility scores"""
import os
import sys
import pickle as pkl
import concurrent.futures
import psutil
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm


class RegressionTester():
    """Testing object for regression"""

    def __init__(self, tr, test_a, test_b=None):
        # convert to pd data frame
        self.regression_model = None
        self.score_fcn = None
        self.train = tr
        self.test_a = test_a
        self.test_b = test_b
        # check dimensions
        assert self.train.shape[1] == self.test_a.shape[1]
        if self.test_b is not None:
            assert self.train.shape[1] == self.test_b.shape[1]
        # preallocate result data frame
        self.scores = pd.DataFrame(
            np.zeros((self.train.shape[1], 1)), columns=['Error Rate'])
        self.scores_b = None
        if self.test_b is not None:
            self.scores_b = pd.DataFrame(
                np.zeros((self.train.shape[1], 1)), columns=['Error Rate'])

        self.regression_model = None
        self.score_fcn = None

    def regression(self, y_idx):
        """overload the parent function"""
        # get training x and y
        train_x = self.train.iloc[:, self.train.columns != self.train.
                                  columns[y_idx]]
        train_y = self.train.iloc[:, y_idx]
        # get test x and y
        test_x = self.test_a.iloc[:, self.train.columns != self.train.
                                  columns[y_idx]]
        test_y = self.test_a.iloc[:, y_idx]
        # if test_b
        if self.test_b is not None:
            test_x_b = self.test_b.iloc[:, self.train.columns != self.train.
                                        columns[y_idx]]
            test_y_b = self.test_b.iloc[:, y_idx]

        # train the model and run prediction
        train_model = self.regression_model()
        train_model.fit(train_x, train_y)
        test_res = self.score_fcn(test_y, train_model.predict(test_x))
        test_res_b = None
        if self.test_b is not None:
            test_res_b = self.score_fcn(test_y_b,
                                        train_model.predict(test_x_b))

        return y_idx, test_res, test_res_b

    def run_test(self):
        """run all segments"""

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(self.regression, i)
                for i in range(self.train.shape[1])
            ]
            # wait for each job to finish
            for idx, future in \
                    tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures)):
                y_idx, err, err_b = future.result()
                self.scores.loc[y_idx] = err
                if err_b:
                    self.scores_b.loc[y_idx] = err_b

        return self.scores, self.scores_b


class LinearRegressionTester(RegressionTester):
    """used for linear regression testing"""

    def __init__(self, *args, **kwargs):
        super(LinearRegressionTester, self).__init__(*args, **kwargs)
        self.regression_model = LinearRegression
        self.score_fcn = r2_score


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(
            'Usage: python nn_adversarial_accuracy.py <prefix_real> <prefix_synth>'
        )
        sys.exit()
    prefix_real = sys.argv[1]
    prefix_synth = sys.argv[2]

    p = psutil.Process()
    p.cpu_affinity(list(range(33, 48)))
    # read in training, testing, and synthetic
    train = pd.read_csv(f'{prefix_real}_train_sdv.csv')
    test = pd.read_csv(f'{prefix_real}_test_sdv.csv')
    synthetics = []
    files = [
        f for f in os.listdir('.') if f.startswith(prefix_synth)
        and f.endswith('.csv') and 'synthetic' in f
    ]
    for f in files:
        synthetics.append(np.clip(pd.read_csv(f), 0, 1))

    # loop through all synthetics
    tr_te_err, _ = LinearRegressionTester(train, test).run_test()
    tr_s_err = []
    te_s_err = []
    for syn in synthetics:
        tr_s, te_s = LinearRegressionTester(syn, train, test).run_test()
        tr_s_err.append(tr_s)
        te_s_err.append(te_s)

    pkl.dump({
        'tr_te_err': tr_te_err,
        'tr_s_err': tr_s_err,
        'te_s_err': te_s_err
    }, open(f'{prefix_synth}_utility.pkl', 'wb'))
