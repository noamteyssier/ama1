#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import sys

sns.set(rc={'figure.figsize':(15, 12), 'lines.linewidth': 5})

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
    def __timeline__(self):
        """generate timelines for each cohortid"""
        self.timelines = self.pr2.pivot_table(
            values = 'c_AveragedFrac',
            index=['cohortid', 'h_popUID'],
            columns='date', dropna=False).\
            dropna(axis=0, how='all')
        self.timelines = (self.timelines > 0).astype('int')
    def __cid_dates__(self):
        """create a series indexed by cid for all dates of that cid"""
        self.cid_dates = self.pr2[['cohortid', 'date']].\
            drop_duplicates().\
            groupby('cohortid').\
            apply(lambda x : x.date.values)
    def __exp_v_obs__(self, x):
        """calculate expected vs observed across timelines by cid"""
        cid = x.name
        haps = [i for _,i in x.index.values]

        # index timeline for cid~dates
        timeline = x.loc[:,self.cid_dates[cid]]#.fillna(0)

        # not enough entries in timeline to call
        if timeline.shape[1] < self.skips:
            return 0, 0

        # prepare arrays for observed and expected
        obs = np.zeros(timeline.shape[1] - self.skips)
        exp = np.zeros(timeline.shape[1] - self.skips)

        # iterate through timelines
        for i in range(self.skips, timeline.shape[1]):
            if self.only_skips:
                true_skip = timeline.iloc[:,i-self.skips + 1:i].values.sum() == 0
            else:
                true_skip=True
            if true_skip:
                # calculate observed
                t_i = timeline.iloc[:,i].values
                t_is = timeline.iloc[:,i-self.skips].values
                obs[i - self.skips] = (t_i + t_is > 1).sum()

                # calculate expected
                t_fs = (self.hapFreq[haps].values * t_is).sum()
                t_f = (t_fs * t_i).sum()
                exp[i - self.skips] = t_f
            else:
                obs[i - self.skips] = 0
                exp[i - self.skips] = 0


        # sum arrays and return
        return exp.sum(), obs.sum()
    def fit(self, s = 0, only_skips=False):
        """run multiple skip evaluations for expected and observed values"""
        self.skips = s + 1
        self.only_skips = only_skips
        eo = self.timelines.\
            groupby(level=0).\
            apply(lambda x : self.__exp_v_obs__(x))

        # pooled numerator and denominator for unweighted estimation
        exp = np.array([i for i,j in eo.values]).sum()
        obs = np.array([j for i,j in eo.values]).sum()

        if obs != 0:
            obs_v_exp = obs.sum() / exp.sum()
        else:
            obs_v_exp = 0

        return obs_v_exp


def main():
    sdo_fn = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta_fn = "../prism2/stata/allVisits.dta"

    sdo = pd.read_csv(sdo_fn, sep='\t')
    meta = pd.read_stata(meta_fn)

    # calculate Expected and Observed for skip vals in range
    eo = ExpObs(sdo, meta)
    skip_vals = np.array([eo.fit(s=i) for i in range(9)])
    only_skip_vals = np.array([eo.fit(s=i, only_skips=True) for i in range(9)])


    # plot EO against number of skips
    p = pd.DataFrame({'skips' : range(9), 'vals' : skip_vals , 'o_vals' : only_skip_vals})
    p = p.melt(id_vars='skips')
    sns.lineplot(data=p, x='skips', y = 'value', hue='variable', style='variable')



if __name__ == '__main__':
    main()
