#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
import os
import sys

from scipy.optimize import minimize
from scipy.special import expit

class Timeline:
    def __init__(self, sdo_fn, meta_fn, event_size, print_histogram):
        self.sdo_fn = sdo_fn
        self.meta_fn = meta_fn
        self.event_size = event_size
        self.print_histogram = print_histogram
        self.model = None

        self.timelines = {}

        self.__parse_sdo__()
        self.__parse_meta__()
        self.__make_timelines__()
        self.__print_histogram__()
    def __parse_sdo__(self):
        """parses seekdeep output for relevant columns"""
        self.sdo = pd.read_csv(self.sdo_fn, sep = "\t")[['s_Sample','h_popUID', 'c_AveragedFrac']]
        self.sdo = self.sdo[self.sdo['s_Sample'].str.contains('ctrl|neg', regex=True) == False] # keep only samples

        # split date and cohortid from s_Sample column
        self.sdo['date'] = self.sdo.\
            apply(lambda x : '-'.join(x['s_Sample'].split('-')[:-1]), axis=1)
        self.sdo['cohortid'] = self.sdo.\
            apply(lambda x : int(x['s_Sample'].split('-')[-1]), axis=1)
    def __parse_meta__(self):
        """parses statabase13 for relevant columns"""
        self.meta = pd.read_stata(self.meta_fn)[['cohortid', 'date', 'ageyrs', 'qpcr', 'visittype', 'malariacat']]

        # apply filters for only cid in sampleset and routine visits except for malaria events
        self.meta = self.meta[self.meta.cohortid.isin(self.sdo.cohortid)]
        self.meta = self.meta[(self.meta.visittype == 'routine visit') | (self.meta.malariacat == 'Malaria')]

        self.meta['date'] = self.meta.\
            apply(lambda x : x.date.strftime('%Y-%m-%d'), axis = 1)

        # create s_Sample column from date and cohortid
        self.meta['s_Sample'] = self.meta.\
            apply(lambda x : '-'.join([x.date, str(x.cohortid)]), axis=1)
    def __make_timelines__(self):
        """create dictionary of cid : haplotype_timelines"""
        self.timelines = {c:self.__haplotype_timelines__(c) for c in self.getPatients()}
    def __haplotype_timelines__(self, cid):
        """create haplotype timelines of all cid~date~haplotype_qpcr_fraction"""
        p_h_timeline = self.getFullTimeline(cid).\
            append(self.getHaplotypeTimeline(cid), ignore_index=False, sort=False).\
            fillna(value=0)
        # patient qpcr values
        qpcr_values = p_h_timeline.values[0]

        # multiply haplotype fractions by qpcr densities
        h_timeline = p_h_timeline[1:].mul(qpcr_values, axis='columns')
        return h_timeline
    def __sliding_window__(self, row, window_size):
        """return a 2d array of sliding windows for a haplotypes infection events"""
        mat = np.array(
            [row.values[i:i+window_size] for i in range(row.values.size-window_size + 1)]
        )
        return mat
    def __age_qpcr_iter_worker__(self, row, cid, n):
        """worker function to split sliding windows and keep age attached"""
        haplotype_windows = []
        for i in range(len(row.index) - n + 1):
            idx = row.index[i:i+n]
            age = self.meta[
                (self.meta.date == idx[0]) &    # age at first event
                (self.meta.cohortid == cid)       # confirm cohortid
            ].ageyrs.values

            haplotype_windows.append((age, row[idx].values.reshape(n,)))
        return haplotype_windows
    def __convert_nlet__(self, arr):
        """convert qpcr data to string of [TF] for greater/equal to 0"""
        return ''.join([i[0] for i in (arr > 0).astype(str)])
    def __fitMS__(self):
        """
        fit MS model on triplets (++*, +-+, +--) and estimate :
            - M = Recovery Probability
            - S = Sensitivity of Detection
        """
        theta = np.random.random(2)
        triplet_list = [self.__convert_nlet__(i) for i in self.timeline_iter(3)]

        self.fit_results = minimize(
            self.__loglikelihood_MS__,
            theta, triplet_list,
            method = 'Nelder-Mead'
        )

        self.__print_fit_results__()
        return self.fit_results
    def __fitAQ__(self):
        theta = np.random.random(4)
        age_triplet_list = [(age, qpcr) for (age, qpcr) in self.age_qpcr_iter(3)]
        self.fit_results = minimize(
            self.__loglikelihood_AQ__,
            theta, age_triplet_list,
            method = 'Nelder-Mead'
        )
        self.__print_fit_results__()
        return self.fit_results

    def __l1__(self, M, S):
        """ likelihood calculation for ++* """
        return np.prod(
                [(1 - M), S]
        )
    def __l2__(self, M, S):
        """ likelihood calculation for +-+ """
        return np.prod(
                [(1-M)**2, (1-S), S]
        )
    def __l3__(self, M, S):
        """ likelihood calculation for +-- """
        return (
            1 - self.__l2__(M,S) - self.__l1__(M,S)
        )
    def __assign_likelihood__(self, triplet, M, S):
        """calculate appropriate likelihood for triplet or return -1"""
        if triplet[:2] == 'TT':
            return self.__l1__(M,S)
        elif triplet == 'TFT':
            return self.__l2__(M,S)
        elif triplet == 'TFF':
            return self.__l3__(M,S)
        else:
            return -1
    def __loglikelihood_MS__(self, theta, triplet_list):
        """calculate log_likelihood for MS triplet model"""
        M,S = theta
        vals = np.array([self.__assign_likelihood__(t, M, S) for t in triplet_list])

        # remove irrelevant triplets from model
        vals = vals[vals != -1]

        # set bound on likelihoods between 0 and 1
        vals[np.where(vals <= 0)] = 1e-6

        # log array and sum
        log_likelihood = np.sum(
            np.log(vals)
        )

        # convert to negative to minimize
        return (-1 * log_likelihood)
    def __loglikelihood_AQ__(self, theta, age_triplet_list):
        """calculate log_likelihood for AQ triplet model with age + qpcr"""
        b0, b1, b2, b3 = theta
        vals = list()
        for age, qpcr in age_triplet_list:
            M = expit(b0 + (b1 * age))
            S = expit(b2 + (b3 * qpcr[0]))
            triplet = self.__convert_nlet__(qpcr)
            l = self.__assign_likelihood__(triplet, M, S)
            vals.append(l)
        vals = np.array(vals)

        # remove irrelevant triplets from model
        vals = vals[vals != -1]

        # set bound on likelihoods between 0 and 1
        vals[np.where(vals <= 0)] = 1e-6

        # log array and sum
        log_likelihood = np.sum(
            np.log(vals)
        )

        # convert to negative to minimize
        return (-1 * log_likelihood)
    def __print_histogram__(self):
        """prints histogram of converted nlets and exits if flag is given"""
        if self.print_histogram:
            triplet_list = [self.__convert_nlet__(i) for i in self.timeline_iter(n = self.event_size)]
            hist = {}
            for t in triplet_list:
                if t not in hist:
                    hist[t] = 0
                hist[t] += 1

            a = []
            b = []
            for t, c in hist.items():
                a.append(t)
                b.append(c)

            self.pattern_hist = pd.DataFrame({'pattern':a, 'count':b})
            self.pattern_hist['frequency'] = self.pattern_hist.\
                apply(lambda x : x['count'] / np.sum(b), axis=1)
            self.pattern_hist.sort_values(by = 'frequency').to_csv(sys.stdout, sep='\t', index=False)

            sys.exit()
    def __print_fit_results__(self):
        """prints fit results as tab delim"""
        print(
            '\t'.join(
                [str(i) for i in self.fit_results.x]
            )
        )
    def getPatients(self):
        """returns a list of unique cohortids found in samples"""
        return self.meta.cohortid.unique()
    def getFullTimeline(self, query_cohortid):
        """return a timeline for a given cohortid of all dates in meta"""
        return(
            self.meta[self.meta.cohortid == query_cohortid].\
            pivot(index = 'cohortid', columns = 'date', values = 'qpcr')
        )
    def getHaplotypeTimeline(self, query_cohortid):
        """return at imeline for a given cohortid of all dates in seekdeep output"""
        return(
            self.sdo[self.sdo.cohortid == query_cohortid].\
            pivot(index = 'h_popUID', columns = 'date', values = 'c_AveragedFrac')
        )
    def timeline_iter(self, n):
        """iterable n-mer sliding window groupings of timelines"""
        for c in self.timelines:
            t = self.timelines[c]
            mat = t.apply(lambda x : self.__sliding_window__(x, n), axis=1)
            try:
                a = np.concatenate(mat, axis = 0).reshape(mat.size, mat[0].shape[0], mat[0].shape[1])
            except IndexError:
                pass
            for haplotype in a:
                for nlet in haplotype:
                    yield nlet
    def age_qpcr_iter(self, n):
        """iterable n-mer sliding window groupings of timelines with ageyrs, qpcr yielded"""
        for cid in self.timelines:
            hap_timelines = self.timelines[cid].\
                apply(lambda x : self.__age_qpcr_iter_worker__(x, cid, n), axis=1)
            for haplotype in hap_timelines.values:
                for age, window in haplotype:
                    yield (age, window)
    def fit(self, model):
        """callable method to fit model of choice"""
        self.model = model
        if model == 'ms':
            return self.__fitMS__()
        elif model == 'aq':
            return self.__fitAQ__()
def get_args():
    """handles arguments, returns args"""
    p = argparse.ArgumentParser()
    p.add_argument("-i", '--seekdeep_input',
        default="../prism2/full_prism2/filtered_5pc_10r.tab",
        help = 'output of SeekDeep to fit model on')
    p.add_argument('-c', '--cohort_meta',
        default="../prism2/stata/allVisits.dta",
        help = '.dta statabase13 meta data for SeekDeep samples')
    p.add_argument('-m', '--model', required=True,
        help = 'model to use [ms, aq]')
    p.add_argument('-n', '--event_size', default=3,
        help = 'event window size to use (default = 3)')
    p.add_argument('-s', '--seed',
        help = 'random seed to use')
    p.add_argument('-p', '--print_histogram', action='store_true',
        help = 'print histogram of n-lets found and exit')
    args = p.parse_args()

    # if no args given print help
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    args.event_size = int(args.event_size)
    return args
def main(args):
    if args.seed:
        np.random.seed(int(args.seed))

    t = Timeline(args.seekdeep_input, args.cohort_meta, args.event_size, args.print_histogram)
    t.fit(args.model)

if __name__ == '__main__':
    args = get_args()
    main(args)
