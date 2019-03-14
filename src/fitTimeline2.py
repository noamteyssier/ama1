#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys, re


class TripletModel:
    def __init__(self, sdo, meta):
        self.sdo_fn = sdo
        self.meta_fn = meta

        self.sdo = pd.DataFrame()
        self.meta = pd.DataFrame()
        self.__load_sdo__()
        self.__load_meta__()
        self.__prepare_triplet_gen__()

        pass
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
        self.meta = pd.read_stata(self.meta_fn)[['cohortid', 'date', 'ageyrs', 'qpcr', 'visittype', 'malariacat']]

        # cid filter
        self.meta = self.meta[self.meta.cohortid.isin(self.sdo.cohortid)]

        # visttype filter
        self.meta = self.meta[(self.meta.visittype == 'routine visit') | (self.meta.malariacat == 'Malaria')]

        # convert to datetime
        self.meta.date = pd.to_datetime(self.meta.date, format='%Y-%m-%d')
    def __window_stack__(self, arr):
        return np.array([arr[i:i+3] for i in range(arr.size-2)])
    def __triplet_iter__(self, x):
        if x.shape[0] >= 3:
            # print(x.qpcr.values)
            windows = self.__window_stack__(x.qpcr.values)
            print(windows)
            # sys.exit()
    def __prepare_triplet_gen__(self):
        a = self.meta.merge(
            self.sdo,
            how='left',
            left_on=['cohortid', 'date'],
            right_on=['cohortid', 'date'])

        sys.exit(a)


        a.groupby(['cohortid', 'h_popUID']).apply(lambda x : self.__triplet_iter__(x))


def main():
    sdo = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta = "../prism2/stata/allVisits.dta"
    TripletModel(sdo, meta)


if __name__ == '__main__':
    main()
