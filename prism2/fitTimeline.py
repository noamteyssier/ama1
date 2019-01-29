#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys

from scipy.optimize import minimize

class Timeline:
    def __init__(self, fn):
        self.df = pd.read_csv(fn, sep = '\t')
        self.cid_arr = None
        self.timelines = []
        self.triplets = []
        self.t_hist = {}
        self.counts = np.array([0, 0, 0]) # L1, L2, L3

        self.__pivot_by_cid__()
        self.__convert_to_triplets__()
        self.__triplet_hist__()
        self.__simplify_triplets__()

    def __pivot_by_cid__(self):
        """add to list timeline dataframe specific to each cohortid"""
        self.cid_arr = self.df.cohortid.unique()
        for c in self.cid_arr:
            cohort_df = self.df[self.df.cohortid == c]
            self.timelines.append(
                cohort_df.pivot(
                    index = 'h_popUID',
                    columns = 'date',
                    values = 'hap_qpcr'
                    ))

    def __convert_to_triplets__(self):
        """convert cid~haplotype events into triplet events"""
        # iterate over each cohortid
        for cid in self.timelines:
            cid_hapsteps = []
            boolArr = np.nan_to_num(cid.values) > 0
            # iterate over haplotype infections
            for haplotype in boolArr:
                hap_steps = []

                # if more than 3 events create sliding windows
                if len(haplotype) > 3:
                    for i in range(len(haplotype) - 2):
                        triplet = ''.join(['T' if i == True else 'F' for i in haplotype[i:i+3]])
                        hap_steps.append(triplet)

                # if just 3 then return a triplet
                elif len(haplotype) == 3:
                    triplet = ''.join(['T' if i == True else 'F' for i in haplotype])
                    hap_steps.append(triplet)

                # add haplotpe to cid
                cid_hapsteps.append(hap_steps)

            # add cid to full set
            self.triplets.append(cid_hapsteps)

    def __triplet_hist__(self):
        """count unique triplet types in dataset"""
        for cid in self.triplets:
            for hid in cid:
                for t in hid:
                    if t not in self.t_hist:
                        self.t_hist[t] = 0
                    self.t_hist[t] += 1

    def __simplify_triplets__(self):
        """simplify triplets for likelihood calculations"""
        for t in self.t_hist:
            if t[:2] == 'TT':
                self.counts[0] += self.t_hist[t]
            elif t == 'TFT':
                self.counts[1] += self.t_hist[t]
            elif t == 'TFF':
                self.counts[2] += self.t_hist[t]

    def fullTripletCount(self):
        """return all unique triplet counts"""
        return self.t_hist

    def tripletCount(self):
        """return unique triplet counts used for likelihood calculations"""
        return self.counts

    def fit_MS(self):
        """optimize M and S fit linearly"""
        theta = np.random.rand(2)
        return minimize(
            self.__calculate_logLikelihood__, theta, method = 'Nelder-Mead'
        )

    def __parameter_limit__(self, param):
        """set bounds of parameter between 1 and 0"""
        if param == 1:
            return param - 1e-5
        elif param == 0:
            return param + 1e-5
        else:
            return param

    def __calculate_logLikelihood__(self, theta):
        """calculate log likelihood as -sum(x_i * log(l_i))"""

        m, s = theta

        l1 = self.__parameter_limit__(
            (1 - m) * s
            )
        l2 = self.__parameter_limit__(
            ((1 - m) ** 2) * (s - s**2)
            )
        l3 = (1 - l2 - l1)

        ind_likelihoods = np.array([l1, l2, l3])
        log_ind_likelihoods = np.log(ind_likelihoods)

        log_likelihood = np.sum(
            log_ind_likelihoods * self.counts
        )

        return (-1 * log_likelihood)







def main():
    os.chdir('/home/noam/bin/ama1/prism2')
    p = Timeline("data/prism2_timeline.tab")
    fit = p.fit_MS()
    print(fit)

if __name__ == '__main__':
    # np.random.seed(42)
    main()
