#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, warnings
sns.set(rc={'figure.figsize':(15, 12), 'lines.linewidth': 5})

class Survival:
    pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    def __init__(self, sdo, meta, burnin='2018-01-01', allowedSkips=3):
        self.sdo = sdo
        self.meta = meta
        self.burnin = pd.to_datetime(burnin)
        self.allowedSkips = allowedSkips

        self.pr2 = pd.DataFrame()
        self.timelines = pd.DataFrame()
        self.cid_dates = pd.Series()
        self.old_new = pd.DataFrame()
        self.date_bins = np.array([])
        self.column_bins = np.array([])

        self.__prepare_df__()
        self.__timeline__()
        self.__cid_dates__()
        self.__label_new_infections__()
        self.__bin_dates__()
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
    def __bin_dates__(self):
        """bin columns (dates) into 12 evenly spaced windows based on first and last visit"""
        dates = pd.to_datetime(self.old_new.columns.values)
        self.date_bins = pd.date_range(dates[0], dates[-1], periods=13)

        counter = 0
        idx = []
        for d in dates:
            if d <= self.date_bins[counter]:
                idx.append(counter)
            else:
                counter += 1
                idx.append(counter)
        idx[0] += 1

        self.column_bins = np.array(idx)
    def __get_skips__(self, x):
        """return skips for timeline row"""
        # find all infection events
        i_event = x.values.nonzero()[1]

        # find distances between infection events
        vals = np.array([
            i_event[i] - i_event[i-1] for i in range(1, i_event.size)])

        # subtract one from distances to reflect true skip size
        return vals - 1
    def __infection_duration__(self, x, skips):
        """split infections into single or multiple dependant on skips"""
        i_event = x.values.nonzero()[1]
        if np.all(skips <= self.allowedSkips):
            # one continuous infection
            return i_event
        else:
            # reinfection occurs under allowed skips
            return np.split(i_event, np.where(skips > self.allowedSkips)[0] + 1)
    def __timeline_marker__(self, timeline, ifx):
        """
        given an infection timeline,
        label old or new if first infection is
        before or after burnin
        1 if old
        2 if new
        """
        t = pd.to_datetime(timeline.columns[ifx[0]])
        ifx = range(ifx.min(), ifx.max() + 1)
        if t <= self.burnin:
            timeline.iloc[:,ifx] = 1
        else:
            timeline.iloc[:,ifx] = 2

        return timeline
    def __type_labeller__(self, x):
        """
        calculate skips
        split infection timelines if reinfection (based on skips)
        label infection timelines as old or new
        return timeline
        """
        cid, hid = x.name
        # if cid != 3839:
        #     return
        timeline = x.loc[:,self.cid_dates[cid]]

        skips = self.__get_skips__(timeline)
        ifx = self.__infection_duration__(timeline, skips)

        timeline[timeline == 0] = np.nan

        # single infection labeller
        if type(ifx) == np.ndarray:
            t = self.__timeline_marker__(timeline, ifx)

        # multiple infection labeller
        if type(ifx) == list:
            [self.__timeline_marker__(timeline, i) for i in ifx]

        return timeline
    def __label_new_infections__(self):
        """create old and new infection timelines"""
        self.old_new = self.timelines.\
            groupby(level=[0,1], sort=False).\
            apply(lambda x : self.__type_labeller__(x))
        self.old_new = self.old_new.fillna(0)
    def OldNewSurvival(self):
        """plot proportion of haplotypes are old v new in population"""
        tw = range(1, self.date_bins.size)

        windows = []
        # iterate through date windows
        for i in tw:
            t = self.old_new.iloc[:,np.where(self.column_bins == i)[0]]
            row_vals = t.apply(lambda x : np.unique(x).sum(), axis = 1)
            vals, count = np.unique(row_vals.values, return_counts=True)
            p = pd.DataFrame({
                'window' : i,
                'type' : vals,
                'counts' : count
            })
            windows.append(p)

        windows = pd.concat(windows)
        self.plot_windows(windows)
    def CID_oldnewsurvival(self):
        """plot proportion of people with old v new v mixed"""
        tw = range(1, self.date_bins.size)

        windows = []
        # iterate through date windows
        for i in tw:
            t = self.old_new.iloc[:,np.where(self.column_bins == i)[0]]
            cid_vals = t.groupby(level=0).apply(lambda x : np.unique(x).sum())

            vals, count = np.unique(cid_vals.values, return_counts=True)

            p = pd.DataFrame({
                'window' : i,
                'type' : vals,
                'counts' : count
            })
            windows.append(p)

        windows = pd.concat(windows)
        self.plot_windows(windows)
    def plot_windows(self, windows):
        date_bins = pd.DataFrame({'t' : range(1, self.date_bins.size), 'bins' : self.date_bins[:-1]})

        # remove zeros
        w = pd.pivot(windows, index='window', columns = 'type', values='counts').drop(columns=0)

        # calculate window sums
        sums = w.sum(axis = 1)

        # normalize windows to sum
        # convert to long
        # get actual window dates
        fracs = w.apply(lambda x : x/sums, axis =0).\
            reset_index().\
            melt(id_vars='window').\
            merge(date_bins, left_on = 'window', right_on='t', how = 'left')
        # convert flaot to int for hue
        fracs.type = fracs.type.astype('int')

        # plot lines
        sns.lineplot(data=fracs, x = 'bins', y = 'value', style='type', hue='type')


def main():
    sdo_fn = "../prism2/full_prism2/filtered_5pc_10r.tab"
    meta_fn = "../prism2/stata/allVisits.dta"

    sdo = pd.read_csv(sdo_fn, sep='\t')
    meta = pd.read_stata(meta_fn)

    # calculate Expected and Observed for skip vals in range
    s = Survival(sdo, meta)
    s.OldNewSurvival()
    s.CID_oldnewsurvival()

if __name__ == '__main__':
    main()
