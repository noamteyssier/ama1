#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.spatial.distance import squareform, pdist

import sys

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
        self.__household_index__()
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
        self.cHap = self.meta.merge(self.cHap, how='left')
    def __household_index__(self):
        """create cohortid-indexed series of household ids"""
        self.hhIndex = self.meta[['cohortid', 'hhid']].\
            drop_duplicates().\
            set_index('cohortid').\
            hhid
    def frequency_of_identity(self, x):
        pv = x[['h_popUID', 'cohortid', 'z']].\
            pivot(
                columns = 'h_popUID',
                index='cohortid',
                values='z'
            ).fillna(0)
        pv = pv.loc[:, pv.columns.notnull()]
        if not pv.empty:
            num = (pv.sum() - 1).sum()
            dem = pv.size
            return num/dem

        return 0

    def clusterHH(self):

        a = self.cHap.groupby('hhid').apply(lambda x : self.frequency_of_identity(x))
        b = self.frequency_of_identity(self.cHap)
        print(a.sum())
        print(b)
        print(a.sum() / b)



def main():
    sdo = '../prism2/full_prism2/filtered_5pc_10r.tab'
    meta = '../prism2/stata/allVisits.dta'
    sc = SpatialClustering(sdo, meta)
    sc.clusterHH()



    pass

if __name__ == '__main__':
    main()
