#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import sys

class Survival:
    pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
    def __init__(self, sdo, meta):
        self.sdo = sdo
        self.meta = meta

        self.__prepare_df__()
    def __prepare_df__(self):
        """prepare dataframe for timeline creation"""
        self.__prepare_sdo__()
        self.__prepare_meta__()
        self.pr2 = self.meta.merge(self.sdo, how='left')
    def __prepare_meta__(self):
        """prepare meta data for usage in timeline generation"""
        self.meta = self.meta[['date', 'cohortid', 'qpcr']]
        self.meta['date'] = self.meta['date'].astype('str')
        self.meta['cohortid'] = self.meta['cohortid'].astype('int')
        self.meta.sort_values(by='date', inplace=True)
        self.meta = self.meta[~self.meta.qpcr.isna()]
    def __prepare_sdo__(self, controls=False):
        """prepare seekdeep output dataframe for internal usage"""
        # keep only patient samples and normalize dataframe
        if controls == False:
            self.sdo = self.sdo[~self.sdo.s_Sample.str.contains('ctrl|neg')]
        else:
            self.sdo = self.sdo

        # split cid and date
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x),
            axis = 1, result_type = 'expand')

        self.sdo['cohortid'] = self.sdo.cohortid.astype('int')

        # select columns of interest
        self.sdo = self.sdo[['cohortid', 'date', 'h_popUID', 'c_AveragedFrac']]
    def __split_cid_date__(self, row):
        """convert s_Sample to date and cohortid"""
        a = row.s_Sample.split('-')
        date, cid = '-'.join(a[:3]), a[-1]
        return [date, cid]
def main():
    sdo_fn = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta_fn = "../prism2/stata/allVisits.dta"

    sdo = pd.read_csv(sdo_fn, sep='\t')
    meta = pd.read_stata(meta_fn)

    # calculate Expected and Observed for skip vals in range
    s = Survival(sdo, meta)
    print(s.pr2)

if __name__ == '__main__':
    main()
