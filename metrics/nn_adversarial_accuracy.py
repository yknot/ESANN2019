'''Compute the nearest neighbor adversarial accuracy'''
import os
import sys
from itertools import product
import pickle as pkl
import concurrent.futures
import psutil
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm


class NearestNeighborMetrics():
    """Calculate nearest neighbors and metrics"""

    def __init__(self, tr, te, synths):
        self.data = {'tr': tr, 'te': te}
        # add all synthetics
        for i, s in enumerate(synths):
            self.data[f'synth_{i}'] = s
        self.synth_keys = [f'synth_{i}' for i in range(len(synths))]
        # pre allocate distances
        self.dists = {}

    def nearest_neighbors(self, t, s):
        """Find nearest neighbors d_ts and d_ss"""
        # fit to S
        nn_s = NearestNeighbors(1).fit(self.data[s])
        if t == s:
            # find distances from s to s
            d = nn_s.kneighbors()[0]
        else:
            # find distances from t to s
            d = nn_s.kneighbors(self.data[t])[0]
        return t, s, d

    def compute_nn(self):
        """run all the nearest neighbors calculations"""
        tasks = product(self.data.keys(), repeat=2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = [
                executor.submit(self.nearest_neighbors, t, s)
                for (t, s) in tasks
            ]
            # wait for each job to finish
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)):
                t, s, d = future.result()
                self.dists[(t, s)] = d

    def divergence(self, t, s):
        """calculate the NN divergence"""
        left = np.mean(np.log(self.dists[(t, s)] / self.dists[(t, t)]))
        right = np.mean(np.log(self.dists[(s, t)] / self.dists[(s, s)]))
        return 0.5 * (left + right)

    def discrepancy_score(self, t, s):
        """calculate the NN discrepancy score"""
        left = np.mean(self.dists[(t, s)])
        right = np.mean(self.dists[(s, t)])
        return 0.5 * (left + right)

    def adversarial_accuracy(self, t, s):
        """calculate the NN adversarial accuracy"""
        left = np.mean(self.dists[(t, s)] > self.dists[(t, t)])
        right = np.mean(self.dists[(s, t)] > self.dists[(s, s)])
        return 0.5 * (left + right)

    def compute_discrepancy(self):
        """compute the standard discrepancy scores"""
        j_rr = self.discrepancy_score('tr', 'te')
        j_ra = []
        j_rat = []
        j_aa = []
        # for all of the synthetic datasets
        for k in self.synth_keys:
            j_ra.append(self.discrepancy_score('tr', k))
            j_rat.append(self.discrepancy_score('te', k))
            # comparison to other synthetics
            for k_2 in self.synth_keys:
                if k != k_2:
                    j_aa.append(self.discrepancy_score(k, k_2))

        # average accross synthetics
        j_ra = np.mean(np.array(j_ra))
        j_rat = np.mean(np.array(j_rat))
        j_aa = np.mean(np.array(j_aa))
        return j_rr, j_ra, j_rat, j_aa

    def compute_divergence(self):
        """compute the standard divergence scores"""
        d_tr_a = []
        d_te_a = []
        for k in self.synth_keys:
            d_tr_a.append(self.divergence('tr', k))
            d_te_a.append(self.divergence('te', k))

        training = np.mean(np.array(d_tr_a))
        testing = np.mean(np.array(d_te_a))
        return training, testing

    def compute_adversarial_accuracy(self):
        """compute the standarad adversarial accuracy scores"""
        a_tr_a = []
        a_te_a = []
        for k in self.synth_keys:
            a_tr_a.append(self.adversarial_accuracy('tr', k))
            a_te_a.append(self.adversarial_accuracy('te', k))

        a_tr = np.mean(np.array(a_tr_a))
        a_te = np.mean(np.array(a_te_a))
        return a_tr, a_te


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

    nnm = NearestNeighborMetrics(train, test, synthetics)
    if f'{prefix_synth}_dists.pkl' in os.listdir('.'):
        dists = pkl.load(open(f'{prefix_synth}_dists.pkl', 'rb'))
        nnm.dists = dists
    else:
        # run all the calculations
        nnm.compute_nn()

    # run discrepancy score, divergence, adversarial accuracy
    discrepancy = nnm.compute_discrepancy()
    divergence = nnm.compute_divergence()
    adversarial = nnm.compute_adversarial_accuracy()

    # save to pickle
    pkl.dump({
        'discrepancy': discrepancy,
        'divergence': divergence,
        'adversarial': adversarial
    }, open(f'{prefix_synth}_results.pkl', 'wb'))

    pkl.dump(nnm.dists, open(f'{prefix_synth}_dists.pkl', 'wb'))
