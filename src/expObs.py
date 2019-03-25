#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

class ExpObs:
    pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
    def __init__(self, sdo, meta):
        self.sdo = sdo
        self.meta = meta

        self.pr2 = pd.DataFrame()
        self.hapFreq = pd.Series()
        self.timelines = pd.DataFrame()
        self.cid_dates = pd.Series()

        self.__prepare_df__()
        self.__haplotype_population_frequency__()
        self.__timeline__()
        self.__cid_dates__()
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
    def __haplotype_population_frequency__(self):
        """determine haplotype frequency in population"""
        self.hapFreq = self.sdo[['cohortid', 'h_popUID']].\
            drop_duplicates().\
            h_popUID.\
            value_counts()
        self.hapFreq = self.hapFreq / self.hapFreq.sum()
    def __pivot_cid__(self, x):
        """pivot cohortid"""
        pv = x.pivot(index = 'h_popUID', columns = 'date', values = 'c_AveragedFrac')
        pv = pv.loc[~pv.index.isna()]
        return pv > 0
    def __timeline__(self):
        """generate timelines for each cohortid"""
        self.timelines = self.pr2.pivot_table(
            values = 'c_AveragedFrac',
            index=['cohortid', 'h_popUID'],
            columns='date')#.reset_index()
    def __cid_dates__(self):
        """create a series indexed by cid for all dates of that cid"""
        self.cid_dates = self.pr2[['cohortid', 'date']].\
            drop_duplicates().\
            groupby('cohortid').\
            apply(lambda x : x.date.values)
    def __exp_v_obs__(self, x):
        cid = x.name
        haps = [i for _,i in x.index.values]
        timeline = x.loc[:,self.cid_dates[cid]]

        print(timeline)

        sys.exit()

    def fit(self, s=0):
        self.skips = s+1
        self.timelines.groupby(level=0).apply(lambda x : self.__exp_v_obs__(x))
        # self.timelines.groupby(['cohortid', 'h_popUID']).apply(lambda x : self.__exp_v_obs__(x))
        # print(self.timelines.loc[3001, self.cid_dates[3001]])




def main():
    sdo_fn = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta_fn = "../prism2/stata/allVisits.dta"

    sdo = pd.read_csv(sdo_fn, sep='\t')
    meta = pd.read_stata(meta_fn)

    eo = ExpObs(sdo, meta)
    eo.fit()

    pass


if __name__ == '__main__':
    main()
