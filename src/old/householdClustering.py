#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.spatial.distance import squareform, pdist
from scipy.stats import percentileofscore
from numpy.random import shuffle
from geopy.distance import distance
from tqdm import tqdm

import sys, timeit

sns.set(rc={'figure.figsize':(15, 12), 'lines.linewidth': 5})
class SpatialClustering:
    def __init__(self, sdo, meta, gps=False):
        self.sdo_fn = sdo
        self.meta_fn = meta
        self.gps_fn = gps

        self.sdo = pd.DataFrame()
        self.meta = pd.DataFrame()
        self.data = pd.DataFrame()
        self.gps = pd.DataFrame()

        self.cHap = pd.DataFrame()
        self.cHapMat = pd.DataFrame()
        self.hhIndex = pd.Series()
        self.pw_dist = pd.DataFrame()
        self.gps_dist = pd.DataFrame()
        self.square_gps_dist = pd.DataFrame()

        self.__load_sdo__()
        self.__load_meta__()
        self.__merge_data__()
        self.__cHap_matrix__()
        self.__pivot_data__()

        if self.gps_fn:
            self.__load_gps__()
            self.__gps_pwdist__()


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
    def __load_gps__(self):
        """load gps as dataframe"""
        self.gps = pd.read_csv(self.gps_fn, sep="\t")
        self.gps = self.gps[self.gps.hhid.isin(self.data.hhid.unique())]
        self.gps['coord'] = self.gps[['lat', 'lng']].apply(tuple, axis=1)
    def __gps_pwdist__(self):
        """pairwise distances of coordinates found in data as squareform and array"""
        self.gps_dist = pdist(self.gps, lambda x,y : distance(x[-1], y[-1]).km)
        self.square_gps_dist = np.tril(squareform(self.gps_dist))
    def __merge_data__(self):
        """merge cohort meta data with seekdeep output"""
        self.data = self.sdo.merge(
            self.meta)
    def __cHap_matrix__(self):
        """create a cohortid~haplotype matrix where 1 is haplotype found in cohortid at any point"""
        self.cHap = self.data[['h_popUID', 'cohortid']].drop_duplicates()
        self.cHap['z'] = 1
        self.cHap = self.meta.merge(self.cHap, how='right')
        self.cHap = self.cHap[self.cHap.z>0]
    def __shuffle_households__(self):
        """shuffle households based on cohortid-hhid pairs"""
        h = self.cHap[['cohortid', 'hhid']].drop_duplicates()
        shuffle(h.hhid.values)
        self.cHap = self.cHap.drop(columns = 'hhid').merge(h)
    def __pivot_data__(self):
        """pivot population data"""
        self.pv = self.cHap[['h_popUID', 'cohortid', 'z']].\
            pivot(
                columns = 'h_popUID',
                index='cohortid',
                values='z'
            ).fillna(0)
    def frequency_of_identity(self, x, rolling=False):
        """Number of matching haplotypes in set divided by number of comparisons"""
        if x.shape[0] > 1:

            # subset population matrix
            cids = x.cohortid.unique()
            hids = x.h_popUID.unique()
            pv = self.pv.loc[cids][hids].values

            if pv.shape[0] > 1:

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

        return [0, 0] if rolling else 0
    def clusterHH(self, shuffle_hhid=False, population=True, rolling=False, simdf=False):
        """
        Pr(Zi == Zj | j in householdSet) /
        Pr(Zi == Zj | j in populationSet)
        """
        params = []

        if shuffle_hhid :
            self.__shuffle_households__()

        # numerator
        a = self.cHap.groupby('hhid').apply(lambda x : self.frequency_of_identity(x, rolling=rolling))

        if rolling:
            # flatten and reshape series of lists
            a = np.array([st for row in a for st in row]).reshape(a.size,2)
            num,dem = a.sum(axis=0)
            a = num/dem

        params.append(a)

        # denominator
        if population:
            b = self.frequency_of_identity(self.cHap)
            print(
                "household similarity : {0}\npopulation similarity : {1}\nratio : {2}".\
                format(a.mean(), b, a.mean()/b))
            params.append(b)

        if simdf:
            s = self.cHap[['cohortid', 'hhid']].drop_duplicates().hhid.value_counts()
            p = pd.DataFrame({'similarity' : a, 'popNum' : s})
            params.append(p)

        return params
    def iHH_counts(self, d):
        """
        given a distance in km (d) return number of comparisons
        that fall within distance for each hh
        """
        a = (self.square_gps_dist <= d) & (self.square_gps_dist > 0)
        b = a.sum(axis=0)
        return b
    def iHH_plot(self, stepDist=False, heatMap=False, boxPlot=False):
        if stepDist:
            sns.distplot(self.gps_dist, kde=False, bins=30)
            plt.xlabel("household distances (km)")
            plt.ylabel("number of pairs")
        elif heatMap:
            sns.heatmap(1 - self.square_gps_dist)
        elif boxPlot:
            lin = np.arange(15) # x axis (distance threshold)
            counts = np.array([self.iHH_counts(i) for i in lin])
            counts = pd.DataFrame(counts)
            counts['lin'] = lin
            melted_counts = pd.melt(a, id_vars='lin', var_name='hh', value_name='comparisons')
            sns.boxplot(x='lin', y='comparisons', data=melted_counts)
        plt.show()


def hhSize_vs_calculatedH(sdo, meta):
    """calculatedH dependence on household size for nonpooled calculation"""
    sc = SpatialClustering(sdo, meta)
    p = [sc.clusterHH(shuffle_hhid=True, population=False, simdf=True)[1] for _ in range(50)]
    comparisons = np.concatenate([i.values for i in p])
    sns.scatterplot(x=comparisons[:,0], y=comparisons[:,1], alpha=0.2, s=100)
    plt.xlabel("Calculated H")
    plt.ylabel("HH size")
    plt.show()
    plt.close()
def pooled_v_average(sdo, meta):
    """distributions of pool/average and the respective calculation for the data"""
    sc = SpatialClustering(sdo, meta)

    # remove outlier
    sc.cHap = sc.cHap[sc.cHap.hhid != 131011801]

    # calculate mean with rolling *erators
    a,b = sc.clusterHH(rolling=False)
    ra,b = sc.clusterHH(rolling=True)

    # run permutations test
    permutations = np.array([sc.clusterHH(shuffle_hhid=True, population=False, rolling=True)[0] for _ in tqdm(range(500), desc='bootstrapping')])
    # o_permutations = np.array([sc.clusterHH(shuffle_hhid=True, population=False)[0] for _ in range(500)])

    # confidence intervals on pooled method
    lower, higher = np.quantile(permutations, [0.025, 1 - 0.025]) / b
    print("CI : {0} - {1}".format(lower, higher))
    print("calculated H : {0}".format(ra/b))
    print("Percentile of H : {}".format(percentileofscore(permutations, ra/b)))

    # plotting
    g = sns.distplot(permutations/b, color='teal', label = 'Permuted Data')
    # sns.distplot(o_permutations.mean(axis=1)/b, color='orange')
    # plt.axvline(a.mean()/b, color='orange')
    plt.axvline(ra/b, color='orange', ls='--', label='Original Dataset')
    plt.legend()
    g.get_figure().savefig("../plots/households/hhCluster.png")
    plt.show()
    plt.close()
def time_analysis(sdo, meta):
    """distribution of time spent in clusterHH"""
    sc = SpatialClustering(sdo, meta)

    # remove outlier
    sc.cHap = sc.cHap[sc.cHap.hhid != 131011801]

    t1 = []
    for i in range(200):
        t = timeit.default_timer()
        sc.clusterHH(shuffle_hhid=True, population=False, rolling=False)
        t1.append(timeit.default_timer() - t)

    t2 = []
    for i in range(200):
        t = timeit.default_timer()
        sc.clusterHH(shuffle_hhid=True, population=False, rolling=True)
        t2.append(timeit.default_timer() - t)

    sns.distplot(np.array(t1))
    sns.distplot(np.array(t2))
    plt.show()



def main():
    fn_sdo = '../prism2/full_prism2/final_filter.tab'
    fn_meta = '../prism2/stata/filtered_visits.dta'
    fn_gps = '../prism2/stata/PRISM_GPS.csv'
    sc = SpatialClustering(fn_sdo, fn_meta, fn_gps)

    # sc.iHH_plot(stepDist=True)
    # a = sc.square_gps_dist.copy()
    # a[a>2] = 0
    # G = nx.from_numpy_matrix(a)
    # nx.draw(G)
    # print(a[:,0])
    #
    # sys.exit()

    # # Time comparisons
    # time_analysis(fn_sdo, fn_meta)
    #
    # # Show variance of calculated H by household size
    # hhSize_vs_calculatedH(fn_sdo, fn_meta)

    # Show difference in pooling vs mean method for permutations with data
    pooled_v_average(fn_sdo, fn_meta)



if __name__ == '__main__':
    main()
