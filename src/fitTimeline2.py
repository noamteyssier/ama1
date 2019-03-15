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

        self.ages = {
            1 : [],
            2 : [],
            3 : [],
            0 : []}
        self.qpcrs = {
            1 : [],
            2 : [],
            3 : [],
            0 : []}

        self.__load_sdo__()
        self.__load_meta__()
        self.__create_likelihood_type_arrays__()
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
        """stack sliding windows as a matrix"""
        return np.array([arr[i:i+3] for i in range(arr.size-2)])
    def __triplet_iter__(self, x):
        """fill missing dates for each h_popUID in a cohortid"""

        # sort by date
        x.sort_values(by = 'date', inplace=True)

        # fill missing dates
        timeline = x.pivot(
                index = 'h_popUID',
                columns = 'date',
                values = ['ageyrs', 'qpcr']).\
            fillna(0)

        # drop na haplotype (artifact of the pivot process)
        timeline = timeline.loc[timeline.index.dropna()]

        # stack sliding windows for qpcr and age
        trip_qpcr = timeline['qpcr'].apply(lambda x : self.__window_stack__(x), axis = 1)
        trip_age = timeline['ageyrs'].apply(lambda x : self.__window_stack__(x), axis = 1)

        return trip_qpcr, trip_age
    def __assign__(self, x):
        """assign triplet array as one of four classes"""
        positives = x>0
        # + + X
        if np.array_equal(positives[:2], np.ones((2,))):
            return 1
        # + - +
        elif np.array_equal(positives, np.array([1,0,1])):
            return 2
        # + - -
        elif np.array_equal(positives, np.array([1,0,0])):
            return 3
        # None
        return 0
    def __assign_triplets__(self, x):
        """assign triplets to dictionary"""
        qpcr_mat, age_mat = x

        # traverse qpcr matrix if haplotype found
        if qpcr_mat.values.shape[0] > 0:

            # for haplotype
            for h_idx in range(qpcr_mat.size):
                # for triplet
                for t_idx in range(qpcr_mat[h_idx].shape[0]):

                    # assign to likelihood class and append to dict
                    likelihood_type = self.__assign__(qpcr_mat[h_idx][t_idx])
                    self.ages[likelihood_type].append(
                        age_mat[h_idx][t_idx][0])
                    self.qpcrs[likelihood_type].append(
                        qpcr_mat[h_idx][t_idx][0])
    def __create_likelihood_type_arrays__(self):
        """
        - merge meta and sdo dataframes
        - create triplets for each haplotype
        - assign triplets to classes
        - grow lists of each class type for age and qpcr
            - take first of each triplet for age/qpcr
        """
        merged_meta_sdo = self.meta.merge(
            self.sdo,
            how='left',
            left_on=['cohortid', 'date'],
            right_on=['cohortid', 'date'])

        # series {cohortid : [qpcr_triplet_matrix, age_triplet_matrix]}
        qpcr_age = merged_meta_sdo.groupby('cohortid').apply(lambda x : self.__triplet_iter__(x))

        # assign triplets to likelihood types and save qpcr and age of each first triplet
        qpcr_age.apply(lambda x : self.__assign_triplets__(x))



def main():
    sdo = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta = "../prism2/stata/allVisits.dta"
    TripletModel(sdo, meta)


if __name__ == '__main__':
    main()
