#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from numpy.random import shuffle

import sys


sns.set(rc={'figure.figsize':(15, 12), 'lines.linewidth': 5})
class SpatialClustering:
    def __init__(self, sdo, meta):
        self.sdo_fn = sdo
        self.meta_fn = meta

        self.sdo = pd.DataFrame()
        self.meta = pd.DataFrame()
        self.data = pd.DataFrame()

        self.cHap = pd.DataFrame()
        self.cHapMat = pd.DataFrame()
        self.hhIndex = pd.Series()
        self.pw_dist = pd.DataFrame()

        self.__load_sdo__()
        self.__load_meta__()
        self.__merge_data__()
        self.__cHap_matrix__()
    def __load_sdo__(self):
        """load in seekdeep output data"""
        self.sdo = pd.read_csv(self.sdo_fn, sep="\t")[['s_Sample', 'h_popUID', 'c_AveragedFrac']]

        # remove controls and negatives
        self.sdo = self.sdo[~self.sdo.s_Sample.str.contains('ctrl|neg')]

        # split cohortid and date
        self.sdo[['date', 'cohortid']] = self.sdo.s_Sample.str.rsplit("-", n = 1, expand=True)

        # convert date to datetime
        self.sdo.date = pd.to_datetime(self.sdo.date, format='%Y-%m-%d')
        self.sdo.cohortid = self.sdo.cohortid.astype('int')

        # drop s_Sample column
        self.sdo = self.sdo.drop(columns = 's_Sample')
    def __load_meta__(self):
        """load in cohort meta data"""
        self.meta = pd.read_stata(self.meta_fn)[['cohortid', 'hhid']].drop_duplicates()
    def __merge_data__(self):
        """merge cohort meta data with seekdeep output"""
        self.data = self.sdo.merge(
            self.meta)
    def __cHap_matrix__(self):
        """create a cohortid~haplotype matrix where 1 is haplotype found in cohortid at any point"""
        self.cHap = self.data[['h_popUID', 'cohortid']].drop_duplicates()
        self.cHap['z'] = 1
        self.cHap = self.meta.merge(self.cHap, how='right')
    def frequency_of_identity(self, x, rolling = False):
        """Number of matching haplotypes in set divided by number of comparisons"""
        if x.shape[0] > 1:
            pv = x[['h_popUID', 'cohortid', 'z']].\
                pivot(
                    columns = 'h_popUID',
                    index='cohortid',
                    values='z'
                ).fillna(0)

            pv = pv.loc[:, pv.columns.notnull()]
            if not pv.empty and pv.shape[0] > 1:

                # sum of (n * (n-1)) / 2 for each haplotype
                colsum = pv.sum(axis=0)
                numerator = ((colsum * (colsum-1)) / 2).sum()

                # sum of (C * (A -C)) for each cohortid
                rowsum = pv.sum(axis=1)
                matsum = np.where(pv > 0)[0].size
                denominator = (rowsum * (matsum - rowsum)).sum()

                if rolling:
                    return [numerator, denominator]

                return numerator / denominator

        return 0
    def clusterHH(self, infOnly=True, shuffle_hhid=False, population=True, rolling=False, simdf=False):
        """
        Pr(Zi == Zj | j in householdSet) /
        Pr(Zi == Zj | j in populationSet)
        """
        params = []

        if infOnly :
            self.cHap = self.cHap[self.cHap.z>0]

        if shuffle_hhid :
            shuffle(self.cHap.hhid.values)


        # numerator
        a = self.cHap.groupby('hhid').apply(lambda x : self.frequency_of_identity(x, rolling=rolling))
        params.append(a)


        if rolling:
            nums = []
            dems = []
            for i in a:
                if type(i) == list:
                    nums.append(i[0])
                    dems.append(i[1])

            nums = np.array(nums)
            dems = np.array(dems)
            a = nums.sum() / dems.sum()

        # denominator
        if population:
            b = self.frequency_of_identity(self.cHap)
            print(
                "household similarity : {0}\npopulation similarity : {1}\nratio : {2}".\
                format(a.mean(), b, a.mean()/b))
            params.append(b)

        if simdf:
            s = self.cHap.hhid.value_counts()
            p = pd.DataFrame({'similarity' : a, 'popNum' : s})
            params.append(p)

        # sys.exit(a)
        return params

def hhSize_vs_calculatedH(sdo, meta):
    sc = SpatialClustering(sdo, meta)
    p = [sc.clusterHH(shuffle_hhid=True, population=False, simdf=True)[1] for _ in range(500)]
    comparisons = np.concatenate([i.values for i in p])
    sns.scatterplot(x=comparisons[:,0], y=comparisons[:,1], alpha=0.2, s=100)
    plt.xlabel("Calculated H")
    plt.ylabel("HH size")
    plt.show()
    plt.close()

def main():
    sdo = '../prism2/full_prism2/filtered_5pc_10r.tab'
    meta = '../prism2/stata/allVisits.dta'
    sc = SpatialClustering(sdo, meta)

    # # Show variance of calculated H by household size
    # hhSize_vs_calculatedH(sdo, meta)

    # remove outlier
    sc.cHap = sc.cHap[sc.cHap.hhid != 131011801]

    # calculate mean with rolling *erators
    a,b = sc.clusterHH()

    sns.distplot(a, bins=30, kde=False)

    # run permutations test
    permutations = np.array([sc.clusterHH(shuffle_hhid=True, population=False, rolling=True)[0] for _ in range(500)])

    # calculate upper and lower bounds of permutation means
    lower, higher = np.quantile(permutations.mean(axis=1), [0.025, 1 - 0.025]) / b
    means = permutations.mean(axis=1)
    means[means > higher].size

    # plot distribution of permutations and observed calculation
    sns.distplot(means/b, bins = 100, kde=True, hist=True)
    plt.axvline(a.mean()/b)




if __name__ == '__main__':
    main()
