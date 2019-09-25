#!/usr/bin/env python3

import numpy as np
import pandas as pd
from tqdm import *
import itertools
import sys
from multiprocessing import *
import seaborn as sns
import matplotlib.pyplot as plt

class HaplotypeUtils:
    def __init__(self, dist, sdo, meta):
        self.dist_fn = dist
        self.sdo_fn = sdo
        self.meta_fn = meta

        self.dist = None # pd.DataFrame()
        self.sdo = None # pd.DataFrame()
        self.meta = None # pd.DataFrame()

        self.melted_dist = None # pd.DataFrame()
        self.one_off = None # pd.DataFrame()
        self.same_sample_pairs = None # pd.DataFrame()
        self.oossp = None # pd.DataFrame()
        self.melted_oossp = None # pd.DataFrame()

        self.__load_dist__()
        self.__load_sdo__()
        self.__load_meta__()
    def __load_dist__(self):
        """read in filename for snp distances into dataframe"""
        self.dist = pd.read_csv(self.dist_fn, sep="\t").\
            rename(columns = {'snp-dists 0.6.3' : 'h_popUID'})
    def __load_sdo__(self):
        """read in filename seekdeep output into dataframe"""
        self.sdo = pd.read_csv(self.sdo_fn, sep="\t")
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x), axis = 1, result_type = 'expand')
    def __load_meta__(self):
        """parses statabase13 for relevant columns"""
        if not self.meta_fn:
            sys.exit('Current Arguments Require Cohort Meta Statabase (-m flag)')

        self.meta = pd.read_stata(self.meta_fn)[['cohortid', 'date', 'qpcr']]

        # convert cohortid to string for joining
        self.meta.cohortid = self.meta.apply(
            lambda x : str(x.cohortid), axis = 1)

        # apply filters for only cid in sampleset and routine visits except for malaria events
        self.meta = self.meta[self.meta.cohortid.isin(self.sdo.cohortid)]

        self.meta.date = self.meta.\
            apply(lambda x : x.date.strftime('%Y-%m-%d'), axis = 1)

        # create s_Sample column from date and cohortid
        self.meta['s_Sample'] = self.meta.\
            apply(lambda x : '-'.join([x.date, str(x.cohortid)]), axis=1)
    def __split_cid_date__(self, row):
        """convert s_Sample to date and cohortid"""
        a = row.s_Sample.split('-')
        date, cid = '-'.join(a[:3]), a[-1]
        return [date, cid]
    def __NaN_upper_triangular__(self, df, colskip=1):
        """
        - convert upper triangular of dataframe to NaN
            - optional column to skip to exclude and reappend after conversion
        """
        skipCol = df.iloc[:, :colskip]
        triangular_bool = np.tril(
            np.ones(df.iloc[:, colskip:].shape).astype(np.bool))
        triangular = df.iloc[:, 1:].where(triangular_bool)
        triangular.insert(
            loc = colskip-1,
            column = skipCol.columns.values[0],
            value = skipCol
        )
        return triangular
    def __melted_dist__(self):
        """creates a long format distance dataframe with steps > 0"""
        self.melted_dist = pd.melt(
            self.__NaN_upper_triangular__(self.dist),
            id_vars=['h_popUID'],
            value_vars=self.dist.columns[1:],
            var_name="h_popUID2",
            value_name="steps"
        )

        # only keep pairs with a distance
        self.melted_dist = self.melted_dist[self.melted_dist.steps > 0]

        # remove population frequency from hid
        self.melted_dist['h_popUID'] = self.melted_dist.\
            apply(lambda x : x['h_popUID'].split('_f')[0], axis = 1)
        self.melted_dist['h_popUID2'] = self.melted_dist.\
            apply(lambda x : x['h_popUID2'].split('_f')[0], axis = 1)

        # rename first h_popUID column
        self.melted_dist.\
            rename(columns = {'h_popUID' : 'h_popUID1'}, inplace=True)
    def __melted_oossp__(self):
        """melted one off same sample pairs dataframe"""
        self.melted_oossp = pd.melt(
            self.oossp.drop(columns = 's_Sample'),
            id_vars=['h_popUID1', 'h_popUID2'],
            value_vars=['h1_fraction', 'h2_fraction'],
            var_name='hap_class',
            value_name='fraction'
        )
        self.melted_oossp.fraction = self.melted_oossp.fraction.astype('float')
        return self.melted_oossp
    def __oneoff__(self):
        """identify one off haplotypes and merge with sdo for h[12] frequencies"""
        self.one_off = self.melted_dist[self.melted_dist.steps == 1]
        self.one_off = self.one_off.\
            merge(
                self.sdo[['h_popUID', 'c_AveragedFrac']],
                left_on = 'h_popUID1',
                right_on = 'h_popUID').\
            merge(
                self.sdo[['h_popUID', 'c_AveragedFrac']],
                left_on = 'h_popUID2',
                right_on = 'h_popUID').\
            rename(columns = {
                'c_AveragedFrac_x' : 'h1_fraction',
                'c_AveragedFrac_y' : 'h2_fraction'}).\
            drop(columns = ['h_popUID_x', 'h_popUID_y', 'steps'])

        # order one_off dataframe so larger uid fraction and haplotype is on the left
        self.one_off = self.__order_haplotype_columns_by_uid__(self.one_off)
    def __samesamplepairs__(self):
        """find haplotype pairs that appear in the same sample"""
        full = [] # holds combinations during iterations

        # iterate through samples and create combinations of all haplotypes found in the sample
        for sample in self.sdo.s_Sample.unique():
            same_sample_haps = self.sdo[self.sdo.s_Sample == sample].h_popUID.values
            if len(same_sample_haps) > 1:
                a = np.array([list(i) for i in itertools.combinations(same_sample_haps, r=2)])
                b = pd.DataFrame({'s_Sample' : sample ,'h_popUID1' : a[:,0], 'h_popUID2' : a[:,1]})
                full.append(b)

        # bind all rows together for one large dataframe
        self.same_sample_pairs = pd.concat(full)

        # merge with sdo to have fraction information with each haplotype in pair
        self.same_sample_pairs = self.same_sample_pairs.\
            merge(
                self.sdo[['s_Sample', 'h_popUID', 'c_AveragedFrac']],
                left_on = ['s_Sample', 'h_popUID1'],
                right_on = ['s_Sample', 'h_popUID']).\
            merge(
                self.sdo[['s_Sample', 'h_popUID', 'c_AveragedFrac']],
                left_on = ['s_Sample', 'h_popUID2'],
                right_on = ['s_Sample', 'h_popUID']).\
            rename(columns = {
                'c_AveragedFrac_x' : 'h1_fraction',
                'c_AveragedFrac_y' : 'h2_fraction'}).\
            drop(columns = ['h_popUID_x', 'h_popUID_y'])

        # order haplotypes and haplotype fractions so larger haplotype uid is on the left
        self.same_sample_pairs = self.__order_haplotype_columns_by_uid__(self.same_sample_pairs)
    def __oossp__(self):
        self.FindOneOff()
        self.FindSameSamplePairs()
        self.oossp = self.one_off.merge(
            self.same_sample_pairs,
            how='inner')
        self.oossp = self.__order_haplotype_columns_by_fraction__(self.oossp)
        self.oossp['majorRatio'] = self.oossp.\
            apply(lambda x : x['h1_fraction'] / x['h2_fraction'], axis = 1)
        self.oossp['minorRatio'] = self.oossp.\
            apply(lambda x : x['h2_fraction'] / x['h1_fraction'], axis = 1)
    def __order_haplotype_columns_by_fraction__(self, df):
        """move major haplotype and related fraction to h1 positions"""
        df.h_popUID1, df.h_popUID2, df.h1_fraction, df.h2_fraction = np.where(
            df.h1_fraction <= df.h2_fraction,
            [df.h_popUID2, df.h_popUID1, df.h2_fraction, df.h1_fraction],
            [df.h_popUID1, df.h_popUID2, df.h1_fraction, df.h2_fraction])
        return df
    def __order_haplotype_columns_by_uid__(self, df):
        """order haplotypes and related fractions so larger uid is in h2 the right"""
        df['uid1'] = df.\
            apply(lambda x : int(x.h_popUID1.split('.')[-1]), axis = 1)
        df['uid2'] = df.\
            apply(lambda x : int(x.h_popUID2.split('.')[-1]), axis = 1)
        df.h_popUID1, df.h_popUID2, df.h1_fraction, df.h2_fraction = np.where(
            df.uid1 <= df.uid2,
            [df.h_popUID2, df.h_popUID1, df.h2_fraction, df.h1_fraction],
            [df.h_popUID1, df.h_popUID2, df.h1_fraction, df.h2_fraction])
        df.drop(columns = ['uid1', 'uid2'], inplace=True)
        return df
    def __haplotype_occurence__(self):
        """create dataframe of haplotype occurences in dataset"""
        self.haplotype_occurences = self.sdo.h_popUID.\
            value_counts()
    def __flag_hvlines__(self, row, vlines, hlines):
        """assign flags to vline and hlines to categorize dataframe for plotting"""
        return_array = []
        for i in vlines:
            if row.h2_fraction > max(vlines):
                return_array.append(">%.2f" % max(vlines))
                break
            if row.h2_fraction < i:
                return_array.append('<%.2f' %i)
                break

        for j in hlines:
            if row.majorRatio > max(hlines):
                return_array.append('>%i' % max(hlines))
                break
            if row.majorRatio < j:
                return_array.append('<%i' %j)
                break
        return return_array
    def __flag_occurence__(self, row):
        """flag occurence of haplotype"""
        oc = self.haplotype_occurences[row.h_popUID2]
        if oc == 1 :
            return '1'
        elif oc == 2:
            return '2'
        elif oc == 3:
            return '3'
        elif oc < 10:
            return '<10'
        else:
            return '>10'
    def __flag_qpcr__(self, qpcr):
        """bin qpcr into log10 densities"""
        if qpcr == 0:
            return '0'
        if int(np.log10(qpcr)) == 1:
            return '1'
        elif (int(np.log10(qpcr))) == 2:
            return '2'
        elif (int(np.log10(qpcr))) == 3:
            return '3'
        elif (int(np.log10(qpcr))) == 4:
            return '4'
        return '5'
    def __normalize_sdo_features__(self, df):
        """normalize seek deep output after applying a filter"""
        sdu = SeekDeepUtils(self.sdo)
        return sdu.fix_filtered_SDO(df)
    def FindOneOff(self):
        """return dataframe of all haplotypes one snp off from each other"""
        self.__melted_dist__()
        self.__oneoff__()
        return self.one_off
    def FindSameSamplePairs(self):
        """return pairs of haplotypes found in same sample"""
        self.__samesamplepairs__()
        return self.same_sample_pairs
    def FindOneOffSameSamplePairs(self, melted=False):
        self.__oossp__()
        if melted:
            return self.__melted_oossp__()
        else:
            return self.oossp
    def PlotOOSSP(self, vlines, hlines, color_type):
        self.__oossp__()
        if color_type == 'fraction':
            self.oossp[['c_Fraction', 'c_Ratio']] = self.oossp.\
                apply(lambda x : self.__flag_hvlines__(x, vlines, hlines),
                result_type = 'expand', axis = 1)

            sns.scatterplot(
                data=self.oossp,
                x='h2_fraction',
                y = 'majorRatio',
                hue = 'c_Fraction',
                s = 100)

            [plt.axvline(i, color = 'darkslategray', linestyle='dashed') for i in vlines]
            [plt.axhline(j, color = 'peru', linestyle='dashed') for j in hlines]
            plt.savefig("../prism2/plots/oossp_fraction.pdf")
        elif color_type == 'occurence':
            self.__haplotype_occurence__()
            self.oossp['h2_occurence'] = self.oossp.apply(
                lambda x : self.__flag_occurence__(x), axis = 1)

            sns.scatterplot(
                data=self.oossp,
                x= 'h2_fraction',
                y = 'majorRatio',
                hue = 'h2_occurence',
                alpha=0.5,
                s = 100)

            [plt.axvline(i, color = 'darkslategray', linestyle='dashed') for i in vlines]
            [plt.axhline(j, color = 'peru', linestyle='dashed') for j in hlines]
            plt.savefig("../prism2/plots/oossp_occurence.pdf")
        elif color_type == 'density':
            self.__load_meta__()
            self.oossp = self.oossp.merge(self.meta[['s_Sample', 'qpcr']])
            self.oossp = self.oossp[self.oossp.qpcr > 0]
            self.oossp['log10_qpcr'] = self.oossp.apply(
                lambda x : np.log10(x.qpcr), axis = 1)

            sns.scatterplot(
                data=self.oossp,
                x='h2_fraction',
                y = 'majorRatio',
                hue = 'log10_qpcr',
                s = 100)

            [plt.axvline(i, color = 'darkslategray', linestyle='dashed') for i in vlines]
            [plt.axhline(j, color = 'peru', linestyle='dashed') for j in hlines]
            plt.savefig("../prism2/plots/oossp_density.pdf")
    def FilterOOSSP(self, ratio = 50, pc = 0.01):
        """apply filter of maj/min pc ratio on one-off haplotypes in the same sample"""
        self.__oossp__()

        # apply filter on oossp dataframe
        self.to_filter = self.oossp[
            (self.oossp.h2_fraction < pc) &
            (self.oossp.majorRatio > ratio)
        ][['h_popUID1', 'h_popUID2', 's_Sample']]

        # melt dataframe for merging with original sdo
        self.to_filter = pd.melt(
            self.to_filter,
            id_vars = 's_Sample',
            value_vars = ['h_popUID1', 'h_popUID2'],
            var_name = 'h_variable',
            value_name = 'h_popUID')

        # merge sdo, drop rows that are found in oossp to be filtered
        self.filtered = self.sdo.merge(
            self.to_filter[self.to_filter.h_variable == 'h_popUID2'],
            how = 'left')
        self.filtered = self.filtered[
            self.filtered.h_variable.isnull()
        ].drop(columns = ['h_variable', 'cohortid', 'date'])

        # recalculate COI, clusterID, Fractions,
        self.filtered = self.__normalize_sdo_features__(self.filtered)

        # print filtered sdo to stdout in proper format
        self.filtered.to_csv(sys.stdout, sep = "\t", index=False)

        return self.filtered
class SeekDeepUtils:
    """class for various utilities related to SeekDeep output"""
    pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
    def __init__(self, sdo, meta=None, fail_flag=True, qpcr_threshold = 0, burnin=3):
        self.sdo = sdo
        self.meta = meta

        # minimum threshold qpcr to consider a sample
        self.qpcr_threshold = qpcr_threshold
        # boolean controlling whether to drop failed sequencing samples
        self.fail_flag = fail_flag
        self.burnin = burnin

        self.__prepare_pr2__()
    def __prepare_pr2__(self):
        self.__prepare_sdo__()
        if type(self.meta) != pd.core.frame.DataFrame:
            return -1
        self.__prepare_meta__()

        # filter qpcr dates
        if self.qpcr_threshold > 0:
            self.meta = self.meta[(self.meta.qpcr == 0) | (self.meta.qpcr >= self.qpcr_threshold)]

        # merge meta and sdo
        self.pr2 = self.meta.merge(self.sdo, how='left')

        # filter all positive qpcr dates with failed sequencing
        if self.fail_flag:
            self.pr2 = self.pr2[~((self.pr2.qpcr > 0) & np.isnan(self.pr2.c_AveragedFrac))]

        self.pr2['h_fraction'] = self.pr2.qpcr * self.pr2.c_AveragedFrac
    def __prepare_meta__(self):
        """prepare meta data for usage in timeline generation"""
        self.meta = self.meta[['date', 'cohortid', 'qpcr', 'agecat', 'enrolldate', 'malariacat', 'ageyrs', 'gender']]
        self.meta['date'] = pd.to_datetime(self.meta['date'])
        self.meta['enrolldate'] = pd.to_datetime(self.meta['enrolldate'])
        self.meta['cohortid'] = self.meta['cohortid'].astype('int')
        self.meta.sort_values(by='date', inplace=True)
        self.meta = self.meta[~self.meta.qpcr.isna()]
        return self.meta
    def __prepare_sdo__(self, controls=False):
        """prepare seekdeep output dataframe for internal usage"""
        # keep only patient samples and normalize dataframe
        if controls == False:
            self.sdo = self.sdo[~self.sdo.s_Sample.str.contains('ctrl|neg')]
            self.sdo = self.fix_filtered_SDO(self.sdo)

        # split cid and date
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x),
            axis = 1, result_type = 'expand')

        self.sdo.date = pd.to_datetime(self.sdo.date)
        self.sdo.cohortid = self.sdo.cohortid.astype('int')

        return self.sdo
    def __prepare_durations__(self, duration_list):
        """melt multiple duration events in dataframe to separate rows and return ordered dataframe"""
        i_durations = pd.concat(duration_list, ignore_index=True)

        # expand listed durations to separate rows
        s = i_durations.apply(
        lambda x: pd.Series(x['durations']),axis=1).\
        stack().\
        reset_index(level=1, drop=True)
        s.name = 'duration'

        # join back on original dataframe
        i_durations = i_durations.\
        drop('durations', axis=1).\
        join(s)

        # sequentially label infection durations
        i_durations['i_event'] = i_durations.\
        groupby(['cohortid', 'h_popUID']).\
        cumcount()

        # convert duration to integer of days
        i_durations['duration'] = i_durations.\
        duration.dt.days

        self.durations = i_durations[['cohortid', 'h_popUID', 'i_event', 'duration']]
        # return ordered dataframe
        return self.durations
    def __prepare_skips__(self):
        """calculates number of skips for each cid~h_popUID at each date"""
        timelines = self.__all_timelines__()

        skip_dataframe = []

        for cid, timeline in timelines.items():

            # case where cid excluded from meta
            if timeline.empty:
                continue

            # find infection events and skips between them
            timeline[['i_events', 'skips']] = timeline.apply(
                lambda x : self.__calculate_skips__(x, diagnose=True),
                axis = 1, result_type = 'expand'
                )

            # for haplotype row create a dataframe of skips and dates
            for _, row in timeline.iterrows():
                skip_dataframe.append(self.__arrange_skips__(row, cid))

        self.skip_df = pd.concat(skip_dataframe)
        self.skip_df.date = pd.to_datetime(self.skip_df.date)

        return self.skip_df
    def __prepare_new_infections__(self, cids, hids, new_ifx):
        """prepare new infections dataframe for downstream usage"""

        # create dataframe from lists
        self.new_infections = pd.DataFrame({
        'cohortid' : cids,
        'h_popUID' : hids,
        'n_infection' : new_ifx})

        # filter haplotypes without new infections
        self.new_infections = self.new_infections[
        self.new_infections.n_infection > 0]

        # sum total new infections by cohortid
        new_infection_totals = self.new_infections.\
        groupby('cohortid').\
        agg({'n_infection' : 'sum'}).\
        rename(columns = {'n_infection' : 'total_n_infection'}).\
        reset_index()

        # merge back into original dataframe
        self.new_infections = self.new_infections.merge(
        new_infection_totals, how = 'left')

        # arrange columns for beauty
        self.new_infections = self.new_infections[
        ['cohortid', 'total_n_infection', 'h_popUID', 'n_infection']]

        return self.new_infections
    def __recalculate_population_fractions__(self):
        self.sdo = self.sdo.\
            groupby('s_Sample').\
            agg({'h_SampFrac' : 'sum'}).\
            rename(columns={'h_SampFrac' : 'h_SampFracSum'}).\
            merge(self.sdo, how = 'left', on = 's_Sample')
        self.sdo['h_SampFrac'] = self.sdo.\
            apply(lambda x : x.h_SampFrac / x.h_SampFracSum, axis = 1)
        self.sdo.drop(['h_SampFracSum'], axis = 1, inplace=True)
    def __recalculate_cluster_fractions__(self):
        """recalculates cluster averaged fraction in sample"""
        self.sdo = self.sdo.\
            groupby('s_Sample').\
            agg({'c_AveragedFrac' : 'sum'}).\
            rename(columns={'c_AveragedFrac' : 's_FracSum'}).\
            merge(self.sdo, how = 'left', on = 's_Sample')
        self.sdo['c_AveragedFrac'] = self.sdo.\
            apply(lambda x : x.c_AveragedFrac / x.s_FracSum, axis = 1)
        self.sdo.drop(['s_FracSum'], axis = 1, inplace=True)
    def __recalculate_cluster_sample_count(self):
        """recalculate number of occurences for a haplotype in population"""
        hapCounts = self.sdo.h_popUID.value_counts()
        self.sdo.h_SampCnt = self.sdo.apply(
            lambda x : hapCounts[x.h_popUID], axis = 1)
    def __reorder_clusterID__(self):
        """recalculates cluster ID number for each sample"""
        self.sdo['c_clusterID'] = self.sdo.\
            groupby('s_Sample').\
            cumcount()
    def __recalculate_COI__(self):
        """recalculates complexity of infection for each sample"""
        self.sdo = self.sdo.\
            groupby('s_Sample').\
            agg({'c_clusterID' : 'max'}).\
            rename(columns = {'c_clusterID' : 'new_COI'}).\
            merge(self.sdo, how = 'left', on = 's_Sample')
        self.sdo['s_COI'] = self.sdo['new_COI'] + 1
        self.sdo.drop('new_COI', axis = 1, inplace = True)
    def __generate_timeline__(self, cohortid, boolArray=False):
        """generate a wideform timeline for a given cohortid"""
        wide_cid = self.pr2[self.pr2.cohortid == cohortid].pivot(
            index = 'h_popUID',
            columns = 'date',
            values = 'h_fraction')

        wide_cid = wide_cid[~wide_cid.index.isna()]

        # returns True if above the given qpcr threshold (default 0)
        if boolArray:
            return wide_cid > self.qpcr_threshold
        else:
            return wide_cid
    def __all_timelines__(self):
        """returns a dictionary of all timelines indexed by cohortid"""
        return {cid : self.__generate_timeline__(cid, boolArray=True) for cid in self.pr2.cohortid.unique()}
    def __infection_labeller__(self, row, allowedSkips):
        """label infections as true or false"""
        # visits before burnin are false
        if row.date <= row.burnin:
            return False
        # first infection occurs at a timepoint past the allowed skips
        elif row.skips > allowedSkips :
            return True
        # if infection is never seen before and after burnin then true
        elif row.skips == row.visit_num - 1:
            return True
        else:
            return False
    def __label_new_infections__(self, allowedSkips):
        """add bool to infection events if they meet skip conditions"""

        # add visit number to meta data to append to skip dataframe
        meta = self.meta[self.meta.cohortid.isin(self.skip_df.cohortid)]
        meta['visit_num']= meta.\
            groupby(['cohortid']).\
            cumcount() + 1

        # merge with visit number
        self.skip_df = self.skip_df.merge(
            meta[['cohortid', 'date', 'visit_num']], how = 'left')

        # merge with cohortid burnin
        self.skip_df = self.skip_df.merge(self.cid_burnin, how='left')

        # label infection event
        self.skip_df['infection_event'] = self.skip_df.apply(
            lambda x : self.__infection_labeller__(x, allowedSkips),
            axis = 1)

        return self.skip_df
    def __calculate_skips__(self, row, diagnose=False):
        """
        calculate skips by subtracting i and i-1 elements of True indices
        input : bool array
        """
        try:
            i_event = row.values.nonzero()[0]
        except AttributeError:
            i_event = row.nonzero()[0]
        vals = np.array([i_event[i] - i_event[i-1] for i in range(1, len(i_event))])
        if diagnose == False:
            vals = vals[vals > 1]
            return vals - 1
        else:
            return (i_event, vals - 1)
    def __generic_skips__(self, row):
        """function to calculate skips in any array"""
        skips = np.array([row[i] - row[i-1] for i in range(1, len(row))])
        return skips - 1
    def __slice_on_skips__(self, skips, allowedSkips):
        """split the indices of infection into separate lists based on the allowed skips"""

        # list of lists initialized with a zero in list[0]
        l = [[0]]

        # counter to keep track of which list is being appended to
        current_list = 0

        # if skip is within allowed range grow current list
        # else create new list and increment counter
        for i,s in enumerate(skips):
            if s > allowedSkips:
                current_list+=1
                l.append([])
            l[current_list].append(i+1)

        return l
    def __arrange_skips__(self, row, cid):
        """create a dataframe from lists created in __calculate_skips__"""
        try:
            # add the position of the initial infection as the first skip
            row.skips = np.insert(row.skips, 0, row.i_events[0])
        except IndexError:
            # case where no infections found above given qpcr threshold
            return None

        # create and return a dataframe
        return pd.DataFrame({
            'cohortid' : cid,
            'h_popUID' : row.name,
            'date' : row.index[:-2].values[row.i_events],
            'skips' : row.skips})
    def __generate_infection_durations__(self, slice_list, i_event, row, default):
        """
        worker function for split infection cases in self.__duration__
        splits infection events based on slice list generated in __slice_on_skips__
        yields either :
        max(dates) - min(dates) for ongoing infections of length longer than 1
            or
        default date for infections of length 1
        """
        dates = [pd.to_datetime(d) for d in row.index]
        idx_gen = [np.array([min(i), max(i)]) if len(i) > 1 else np.array(i) for i in slice_list]
        idx = [i_event[i] for i in idx_gen]
        for i in idx:
            if len(i) > 1:
                yield dates[max(i)] - dates[min(i)]
            else:
                yield default
    def __duration__(self, row, allowedSkips, default=15):
        """calculate the durations of a haplotype given alllowed skips in events"""
        default = pd.to_timedelta(default, unit='day')
        skips = self.__calculate_skips__(row)
        if len(skips) == 0:
            """
            case where only a single event is recorded for a haplotype
            ::::::::::::::::::::::::::::
            return default duration rate
            """
            return default
        if np.any(skips > allowedSkips):
            i_event, skips = self.__calculate_skips__(row, diagnose=True)
            if np.all(skips > allowedSkips):
                """
                case where all events are outside allowed skips
                :::::::::::::::::::::::::::::::::::::::::::::::::::::
                return default duration rate for each infection event
                """
                return [default for _ in i_event]
            else:
                """
                case where some events are outside allowed skips
                :::::::::::::::::::::::::
                return array of durations
                """

                l = self.__slice_on_skips__(skips, allowedSkips)
                durations = [d for d in self.__generate_infection_durations__(l, i_event, row, default)]
                return durations
        else:
            """
            default case, where all events are within allowed skips
            :::::::::::::::::::
            return last - first
            """
            idx = np.where(row > 0)[0]
            end, start = [pd.to_datetime(i) for i in [row.index[max(idx)], row.index[min(idx)]]]
            duration = end - start
            return end - start
    def __haplotype_infections__(self, dates, allowedSkips):
        """calculate number of new infection events for a given haplotype"""
        skips = self.__generic_skips__(dates)

        # if all skips are allowed then only a single infection is recorded
        if np.all(skips <= allowedSkips):
            n_infections = 1

        # otherwise slice the skips into separate events and count them
        else:
            sliced_skips = self.__slice_on_skips__(skips, allowedSkips)
            n_infections = len(sliced_skips)

        # if the first date is the first visit subtract 1 from n_infections
        if min(dates) == 0:
            n_infections -= 1

        return n_infections
    def __split_cid_date__(self, row):
        """convert s_Sample to date and cohortid"""
        a = row.s_Sample.split('-')
        date, cid = '-'.join(a[:3]), a[-1]
        return [date, cid]
    def __add_agecat_to_sdo__(self):
        """add agecategories to seekdeep dataframe"""
        self.age_categories = self.meta.\
            groupby('cohortid').\
            tail(1)[['cohortid', 'agecat']]

        self.sdo = self.sdo.merge(self.age_categories, how='left')
    def __foi_exposure__(self, meta, agecat=False, month=False):
        """calculate exposure for a given sdo dataframe"""
        if agecat:
            exposure = self.age_categories.\
                groupby('agecat').\
                size().\
                reset_index().\
                rename(columns = {0 : 'exposure'})
        else:
            exposure = self.meta['cohortid'].drop_duplicates().size

        return exposure
    def __foi_exposure_duration__(self, sdo, agecat=False):
        """calculate duration in years for a given sdo dataframe"""
        sdo.date = pd.to_datetime(sdo.date, format = '%Y/%m/%d')
        if agecat:
            diff = sdo.\
                groupby('agecat').\
                date.\
                apply(lambda x : x.max() - x.min()).\
                reset_index().\
                rename(columns = {'date' : 'duration'})
            diff.duration = diff.\
                apply(lambda x : x.duration.days / 365.25, axis = 1)
        else:
            diff = (sdo.date.max() - sdo.date.min()).days / 365.25

        return diff
    def __foi_collapse_infection_by_person__(self):
        """collapse infections of a person_datetime to a single value"""
        personal_infection = self.sdo.\
            groupby('s_Sample').\
            agg({'infection_event' : 'max'})
        self.sdo['person_infection'] = self.sdo.apply(
            lambda x : personal_infection.infection_event[x.s_Sample], axis = 1)
        self.sdo = self.sdo[
            ['s_Sample', 'date', 'cohortid', 'visit_num', 'person_infection']].\
            drop_duplicates().\
            rename(columns = {'person_infection' : 'infection_event'})
    def __foi_cid_duration_from_first_visit__(self):
        """find duration of each cid from first visit to last visit in years"""

        first_visits = self.meta[self.meta.date >= self.meta.burnin].\
            groupby('cohortid').\
            head(1)[['cohortid', 'date']].\
            rename(columns = {'date' : 'first_visit'})


        last_visits = self.meta.\
            groupby('cohortid').\
            tail(1)[['cohortid', 'date']].\
            rename(columns = {'date' : 'last_visit'})

        self.cid_dates = first_visits.\
            merge(last_visits, how = 'left')
        self.cid_dates['first_visit'] = pd.to_datetime(
            self.cid_dates['first_visit'], format='%Y-%m-%d')
        self.cid_dates['last_visit'] = pd.to_datetime(
            self.cid_dates['last_visit'], format='%Y-%m-%d')


        self.cid_dates['duration'] = self.cid_dates.apply(
            lambda x : (x.last_visit - x.first_visit).days / 365.25,
            axis = 1)

        return self.cid_dates
    def __foi_method_all__(self, individual=False, agecat=False):
        """calculate force of infection over the entire dataset"""

        # apply burnin to sdo
        self.sdo = self.sdo[self.sdo.date >= self.sdo.burnin]

        if individual == True:
            self.__foi_collapse_infection_by_person__()

        if agecat == True:
            self.__add_agecat_to_sdo__()
            duration = self.__foi_exposure_duration__(self.sdo, agecat)
            exposure = self.__foi_exposure__(self.meta, agecat)
            new_infections = self.sdo.\
                groupby('agecat').\
                infection_event.\
                sum().\
                reset_index()
            foi = new_infections.\
                merge(duration).\
                merge(exposure)
            foi['force_of_infection'] = foi.apply(
                lambda x : x.infection_event / (x.duration * x.exposure),
                axis = 1)
        else:
            new_infections = self.sdo.infection_event.sum()
            duration = self.__foi_exposure_duration__(self.sdo, agecat)
            exposure = self.__foi_exposure__(self.meta, agecat)
            foi_scalar = new_infections / (duration * exposure)
            foi = pd.DataFrame({
                'infection_event' : [new_infections],
                'duration' : [duration],
                'exposure' : [exposure],
                'force_of_infection' : [foi_scalar]})

        return foi
    def __foi_method_month__(self, individual=False, agecat=False):
        """calculate force of infection by month and by clone"""
        relevant_columns = ['cohortid', 'date', 'h_popUID', 'infection_event']
        groupby_columns = ['ym']

        # apply burnin to sdo
        self.sdo = self.sdo[self.sdo.date >= self.sdo.burnin]

        # collapse all clones to a single infection event at a date
        if individual == True:
            self.__foi_collapse_infection_by_person__()
            relevant_columns = ['cohortid', 'date', 'infection_event']

        # collapse infections
        if agecat == True:
            self.__add_agecat_to_sdo__()
            groupby_columns = ['ym', 'agecat']

        # merge with all cohortids for accurate population FOI
        self.sdo = self.sdo[relevant_columns].\
            merge(self.meta, how = 'right').\
            fillna(value={'infection_event' : False})

        # groupby month (and optionally agecat) for infection event sum
        self.sdo['ym'] = self.sdo.date.dt.to_period('M')
        monthly_infections = self.sdo.\
            groupby(groupby_columns).agg({
                'infection_event' : 'sum',
                'date' : lambda x : (max(x) - min(x)).days / 365.25
                }).\
            reset_index()


        # get exposure by group
        if not agecat:
            monthly_infections['exposure'] = monthly_infections.apply(
                lambda x : self.meta[self.meta.burnin <= x.ym.to_timestamp() + pd.offsets.Day(28)].cohortid.unique().size,
                axis=1)
        else:
            monthly_infections['exposure'] = monthly_infections.apply(
                lambda x : self.meta[
                    (self.meta.agecat == x.agecat) &
                    (self.meta.burnin <= x.ym.to_timestamp() + pd.offsets.Day(28))
                    ].cohortid.unique().size,
                axis=1)

        # calculate FOI
        monthly_infections['foi'] = monthly_infections.apply(
            lambda x : x.infection_event / (x.date * x.exposure) if x.exposure > 0 else 0,
            axis=1)
        return monthly_infections
    def __foi_method_person__(self, individual=False):
        """calculate force of infection for each person"""

        # apply burn in to dataframe
        self.sdo = self.sdo[self.sdo.date >= self.sdo.burnin]

        # calculate infections with collapsed haplotype infection events
        if individual == True:
            self.__foi_collapse_infection_by_person__()

        # calculate infection events by cohortid
        cid_infections = self.sdo.\
            groupby(['cohortid']).agg({
                'infection_event' : 'sum',
            }).\
            reset_index()

        # merge cid dates with cid infections and fill single infections to 0
        self.cid_dates = self.__foi_cid_duration_from_first_visit__()
        cid_infections = cid_infections.\
            merge(self.cid_dates, how='left').\
            drop(columns = ['first_visit', 'last_visit']).\
            fillna(0)

        # calculate force of infection
        cid_infections['foi'] = cid_infections.apply(
            lambda x : x.infection_event / x.duration if x.duration > 0 else 0, axis = 1)

        return cid_infections
    def __label_haplotype_infection_type__(self, group):
        """
        for each new infection past the burnin :
        assign all following haplotype events as new infections
        """
        # find infection events past the burnin

        group['true_new'] = group.apply(
            lambda x : (x.infection_event) & (x.date >= x.burnin),
            axis = 1)

        # find if there are any new infections past the burnin
        true_news = np.where(group.true_new.values > 0)[0]

        # if so, label all following haplotype events as so
        if len(true_news) > 0:
            group.iloc[true_news.min():, -1] = True

        return group
    def FindCIDBurnin(self, month_offset=3):
        """find burnin period for all cohortids given a month offset"""
        # get enrolldate by cid
        cid_burnin = self.pr2.\
            groupby('cohortid').\
            apply(lambda x : x.enrolldate.min()).\
            reset_index().\
            rename(columns={0 : 'enrolldate'})
        # shift by burnin
        cid_burnin['burnin'] = cid_burnin.\
            apply(lambda x: x['enrolldate'] + pd.DateOffset(months = self.burnin), axis=1)
        self.meta['burnin'] = self.meta.\
            apply(lambda x: x['enrolldate'] + pd.DateOffset(months = self.burnin), axis=1)

        return cid_burnin
    def fix_filtered_SDO(self, sdo):
        """recalculates attributes of SeekDeep output dataframe post-filtering"""
        self.sdo = sdo
        if not self.sdo.empty:
            self.__recalculate_population_fractions__()
            self.__recalculate_cluster_fractions__()
            self.__reorder_clusterID__()
            self.__recalculate_COI__()
            self.__recalculate_cluster_sample_count()
        return self.sdo
    def Time_Independent_Allele_Frequency(self, sdo, controls=False):
        """calculates allele frequency of each haplotype in population with independent time"""

        # initialize dataframe to return
        a_freq = self.sdo[['h_popUID']].drop_duplicates()

        # select h_popUID + cid and drop duplicates created by dates
        hapCounts = self.sdo[['h_popUID', 'cohortid']].\
            drop_duplicates().\
            h_popUID.\
            value_counts()
        hapCountsTotal = sum(hapCounts)

        # calculate h_popUID counts and frequencies (time independent)
        a_freq['h_Count'] = a_freq.apply(
            lambda x : hapCounts[x.h_popUID], axis=1)
        a_freq['h_Frequency'] = a_freq.apply(
            lambda x : x.h_Count / hapCountsTotal, axis = 1)

        return a_freq
    def Haplotype_Skips(self, controls=False):
        """returns an array of the skips found per cid~haplotype across dates"""
        self.__prepare_skips__()
        return self.skip_df
    def Duration_of_Infection(self, sdo, meta, controls=False, allowedSkips = 3, default=15):
        """calculates duration of infections for each cohortid ~ h_popUID"""

        # generate timelines for each cohortid~h_popUID
        timelines = {c : self.__generate_timeline__(c, boolArray=True) for c in self.sdo.cohortid.unique()}

        # list of duration dataframes to grow
        i_durations = []

        # calculate duration of infections for each cohortid~h_popUID
        for cid, t in timelines.items():
            # calculate durations
            t['durations'] = t.apply(
                lambda x : self.__duration__(x, allowedSkips, default=default), axis = 1)

            # add cohortid as column of dataframe
            t['cohortid'] = cid

            # grow list of duration dataframes
            i_durations.append(t.reset_index()[['cohortid', 'h_popUID', 'durations']])

        # concatenate all dataframes and prepare for downstream susage
        self.durations = self.__prepare_durations__(i_durations)

        return self.durations
    def Old_New_Infection_Labels(self, controls=False, allowedSkips = 3, default=15):
        """labels cid~hid infections that developed past a burn-in date as new else old"""
        self.cid_burnin = self.FindCIDBurnin(month_offset=3)
        self.skip_df = self.__prepare_skips__()
        self.__label_new_infections__(allowedSkips)

        hit_labels = self.skip_df.\
            groupby(['cohortid', 'h_popUID']).\
            apply(lambda x : self.__label_haplotype_infection_type__(x))

        return hit_labels
    def New_Infections(self, controls=False, allowedSkips = 3, burnin='2018-01-01'):
        """calculates number of new infections for each haplotype in each cohortid with allowed skips"""
        # self.burnin = pd.to_datetime(burnin)
        self.cid_burnin = self.FindCIDBurnin(month_offset=3)
        self.__prepare_skips__()
        self.__label_new_infections__(allowedSkips)

        return self.skip_df
    def Force_of_Infection(self, controls=False, foi_method = 'all', allowedSkips = 3, default=15, burnin=3):
        """calculate force of infection for a dataset"""
        self.burnin = burnin
        self.cid_burnin = self.FindCIDBurnin(self.burnin)
        self.__prepare_skips__()
        self.__label_new_infections__(allowedSkips)
        self.sdo = self.sdo.merge(self.skip_df, how='left')

        if foi_method == 'all':
            return self.__foi_method_all__()
        elif foi_method == 'all_individual':
            return self.__foi_method_all__(individual=True)
        elif foi_method == 'all_individual_agecat':
            return self.__foi_method_all__(individual=True, agecat=True)
        elif foi_method == 'all_agecat':
            return self.__foi_method_all__(agecat=True)


        elif foi_method == 'month':
            return self.__foi_method_month__()
        elif foi_method == 'month_individual':
            return self.__foi_method_month__(individual=True)
        elif foi_method == 'month_agecat':
            return self.__foi_method_month__(agecat=True)
        elif foi_method == 'month_individual_agecat':
            return self.__foi_method_month__(individual=True, agecat=True)
        elif foi_method == 'cid':
            return self.__foi_method_person__(individual=False)
        elif foi_method == 'cid_individual':
            return self.__foi_method_person__(individual=True)
        else:
            sys.exit('Error : FOI method "{0}" not recognized'.format(foi_method))


class InfectionLabeler:
    """
    Label Infection events given a qpcr threshold, burnin period, and allowed skips
    """
    def __init__(self, sdo, meta, qpcr_threshold = 0, burnin=3, allowedSkips = 3, by_individual=False, impute_missing=False):

        self.sdo = sdo
        self.meta = meta
        self.qpcr_threshold = qpcr_threshold
        self.burnin = burnin
        self.allowedSkips = allowedSkips
        self.by_individual = by_individual
        self.impute_missing = impute_missing
        self.is_bootstrap = False

        # post processed sdo + meta
        self.frame = pd.DataFrame()

        # annotated skips dataframe
        self.skips = pd.DataFrame()

        # annotated infection events dataframe
        self.labels = pd.DataFrame()

        self.InitFrames()
    def PrepareSDO(self):
        """
        - Split cohortid and date from sample name
        - Convert dates to pd.datetime
        - Convert cohortid to int
        """
        split_date_cid = np.vstack(self.sdo.s_Sample.str.split('-'))

        # split date and cid from s_Sample
        self.sdo['date'] = pd.to_datetime(['-'.join(date) for date in split_date_cid[:,:3]])
        self.sdo['cohortid'] = split_date_cid[:,-1].astype(int)
    def PrepareMeta(self):
        """
        - Convert date to pd.datetime
        - Convert enrolldate to pd.datetime
        - Convert cohortid to int
        - Sort values by date
        - Add burnin date
        """

        self.meta['date'] = pd.to_datetime(self.meta['date'])
        self.meta['enrolldate'] = pd.to_datetime(self.meta['enrolldate'])
        self.meta['cohortid'] = self.meta['cohortid'].astype(int)

        self.meta.sort_values(['cohortid', 'date'], inplace=True)

        return self.meta
    def MarkQPCR(self):
        """
        Keep samples above qpcr threshold (removes NaN as well)
        """
        self.meta['pass_qpcr'] = self.meta.qpcr >= self.qpcr_threshold
    def MergeFrames(self):
        """
        - Merge seekdeep output and meta data
        - Update h_popUID type
        """
        self.frame = self.meta.merge(
            self.sdo, how = 'left',
            left_on = ['cohortid', 'date'],
            right_on = ['cohortid', 'date']
            )
        self.frame['h_popUID'] = self.frame['h_popUID'].astype(str)
        self.frame.date = self.frame.date

        if 'pseudo_cid' in self.frame.columns:
            self.is_bootstrap = True
    def AddBurnin(self):
        """
        Generate burnin and add to Meta

        - Take minimum enrolldate per cohortid
        - Add {burnin} months to enrolldate
        - Merge with meta frame
        """

        cid_enroll = self.meta.\
            groupby('cohortid').\
            agg({'enrolldate' : 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + pd.DateOffset(months = self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on = ['cohortid', 'enrolldate'],
            right_on = ['cohortid', 'enrolldate']
            )
    def InitFrames(self):
        """
        - Initialize Seekdeep Output & Meta
        - Add burnin dates to Meta
        - Mark QPCR passing threshold
        - Merge
        """
        self.PrepareSDO()
        self.PrepareMeta()
        self.AddBurnin()
        self.MarkQPCR()
        self.MergeFrames()
        self.LabelSkips()
    def positionalDifference(self, x, post_burnin):
        """
        For a bool array x : calculate number of skips between each truth
        calculate first truth with respect to the number of visits post burnin
        """
        truth = np.where(x)[0]

        if truth.size == 0:
            return np.array([])

        skip_arr = []
        for i, pos in enumerate(truth):

            # first occurence of a haplotype
            if i == 0:

                # switch for burnin period
                if pos < post_burnin:
                    skips = pos
                else:
                    skips = pos - post_burnin


            else:

                # disregard pre-burnin skips
                if (truth[i - 1] < post_burnin) & (pos >= post_burnin):
                    skips = pos - post_burnin

                # calculate skips in pre-burnin period
                else:
                    skips = pos - truth[i-1] - 1

            skip_arr.append(skips)

        return np.array(skip_arr)
    def SkipsByHaplotype(self, hid_timelines, cid, post_burnin):
        """
        Calculate Skips for each haplotype given an HID_timeline
        """
        # get positional difference between all passing qpcr events
        for hid, hid_frame in hid_timelines.groupby('h_popUID'):

            skips = self.positionalDifference(hid_frame.values[0], post_burnin)
            dates = hid_frame.columns[hid_frame.values[0]]
            visit_numbers = np.where(hid_frame.columns.isin(dates))[0]

            if skips.size > 0:
                self.skip_frame.append(
                    [[cid, hid, dates[i], skips[i], visit_numbers[i]]
                    for i,_ in enumerate(skips)]
                )
    def SkipsByIndividual(self, hid_timelines, cid, post_burnin):
        """
        Calculate Skips for an individual given an HID_timeline
        """
        if hid_timelines.shape[0] == 0 :
            return
        timeline = hid_timelines.values.max(axis=0)
        skips = self.positionalDifference(timeline, post_burnin)
        cid_dates = hid_timelines.columns[timeline]
        visit_numbers = np.where(timeline)[0]

        if skips.size > 0:
            self.skip_frame.append(
                [[cid, 'agg_{}'.format(cid), cid_dates[i], skips[i], visit_numbers[i]]
                for i,_ in enumerate(skips)]
            )
    def LabelSkips(self):
        """
        - Build timelines for each h_popUID~cohortid combination
        - Calculate number of skips between each passing qpcr event
        - Calculate visit number of events
        - Compile dataframe of skips and visit number
        """

        self.skip_frame = []
        for cid, cid_frame in tqdm(self.frame.groupby(['cohortid']), desc='calculating skips'):
            burnin = cid_frame.burnin.values[0]

            # convert long dates to wide
            cid_timeline = cid_frame.pivot_table(
                index = 'h_popUID',
                columns = 'date',
                values = 'pass_qpcr'
                )

            try:
                post_burnin = np.where(cid_timeline.columns >= burnin)[0].min()
            except ValueError:
                post_burnin = False

            # remove NaN h_popUID and fill pass_qpcr with False
            hid_timelines = cid_timeline[cid_timeline.index != 'nan'].\
                fillna(False)

            if self.by_individual:
                self.SkipsByIndividual(hid_timelines, cid, post_burnin)
            else:
                self.SkipsByHaplotype(hid_timelines, cid, post_burnin)

        # stack events, and build dataframe
        self.skips = pd.DataFrame(
            np.vstack(self.skip_frame),
            columns = ['cohortid', 'h_popUID', 'date', 'skips', 'visit_number']
            )

        self.skips.date = pd.to_datetime(self.skips.date)
    def getLabel(self, row):
        """label infections as true or false"""

        # visits before burnin are false
        if row.date <= row.burnin:
            return False

        # first infection occurs at a timepoint past the allowed skips
        elif row.skips > self.allowedSkips :
            return True

        # if infection is never seen before and after burnin then true
        elif row.skips == row.visit_number:
            return True

        else:
            return False
    def ActiveInfection(self, group):
        """
        Label all timepoints where an infection is still active
        """

        infections = np.where(group.infection_event)[0]

        active_array = np.zeros(group.infection_event.size)

        # for each infection event, label all following infections as active
        for i in np.arange(infections.size):
            if i < infections.size - 1:
                active_array[infections[i] : infections[i+1]] = 1
            else:
                active_array[infections[i] : ] = 1


        return active_array
    def getColumns(self):
        """
        Returns columns to merge on and columns to keep from frame
        """
        merging = ['cohortid', 'date']
        to_keep = ['cohortid', 'date', 'enrolldate', 'burnin']
        if not self.by_individual:
            merging.append('h_popUID')
            to_keep.append('h_popUID')
        if self.is_bootstrap:
            to_keep.append('pseudo_cid')
        return merging, to_keep
    def LabelInfections(self):
        """
        Label timepoints as infection events
        &
        Label subsequence timepoints of infection events as active infections
        """

        merging, to_keep = self.getColumns()

        self.labels = self.skips.merge(
            self.frame[to_keep],
            left_on = merging,
            right_on = merging,
            how='inner'
            ).drop_duplicates()

        self.labels['infection_event'] = self.labels.apply(
            lambda x : self.getLabel(x),
            axis = 1
            )

        active_id = 'pseudo_cid' if self.is_bootstrap else 'cohortid'
        self.labels['active_infection'] = np.concatenate(self.labels.groupby([active_id, 'h_popUID']).apply(
            lambda x : self.ActiveInfection(x).astype(bool)
        ).values)

        return self.labels
class FOI:
    def __init__(self, labels, meta, burnin=3):
        self.labels = labels
        self.meta = meta
        self.burnin = burnin

        self.frame = pd.DataFrame()
        self.is_bootstrap = False

        self.prepareData()
    def prepareData(self):
        """
        validate column types
        add burnin to meta
        merge labels with meta
        """
        self.labels.date = pd.to_datetime(self.labels.date)
        self.labels.enrolldate = pd.to_datetime(self.labels.enrolldate)
        self.labels.burnin = pd.to_datetime(self.labels.burnin)

        self.meta.date = pd.to_datetime(self.meta.date)
        self.meta.enrolldate = pd.to_datetime(self.meta.enrolldate)

        self.AddBurnin()

        if 'pseudo_cid' in self.meta.columns:
            self.is_bootstrap = True
            self.labels = self.labels.drop(columns = 'pseudo_cid')

        self.frame = self.meta.merge(
            self.labels,
            left_on = ['cohortid', 'date', 'enrolldate', 'burnin'],
            right_on = ['cohortid', 'date', 'enrolldate', 'burnin'],
            how = 'left'
            )

        # check if dataframe is a bootstrap, and use pseudo_cid instead of cid post merging
        if 'pseudo_cid' in self.frame.columns:
            self.is_bootstrap = True
            self.frame['pseudo_cid'] = self.frame.apply(
                lambda x : str(x.cohortid) + '_0' if str(x.pseudo_cid) == 'nan' else x.pseudo_cid,
                axis=1
            )
    def AddBurnin(self):
        """
        Generate burnin and add to Meta

        - Take minimum enrolldate per cohortid
        - Add {burnin} months to enrolldate
        - Merge with meta frame
        """

        cid_enroll = self.meta.\
            groupby('cohortid').\
            agg({'enrolldate' : 'min'}).\
            reset_index()

        cid_enroll['burnin'] = cid_enroll['enrolldate'] + pd.DateOffset(months = self.burnin)

        self.meta = self.meta.merge(
            cid_enroll,
            left_on = ['cohortid', 'enrolldate'],
            right_on = ['cohortid', 'enrolldate']
            )
    def getDurations(self, group=None, working_frame=None):
        """
        return durations across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()

        working_frame = working_frame[working_frame.date >= working_frame.burnin]

        if group:
            durations = working_frame.\
                groupby(group).\
                apply(lambda x : self.getDurations(working_frame=x))

        else:
            durations = working_frame.date.max() - working_frame.date.min()
            durations = durations.days / 365.25

        return durations
    def getInfections(self, group=None, working_frame=None):
        """
        return number of infections across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()
        working_frame = working_frame[working_frame.date >= working_frame.burnin]

        if group:
            events = working_frame.\
                groupby(group).\
                apply(lambda x : self.getInfections(working_frame=x))
        else:
            events = working_frame.infection_event.sum()

        return events
    def getExposure(self, group=None, working_frame=None):
        """
        return number of exposed individuals across a group or a singular value for the full dataset
        """
        if type(working_frame) == type(None):
            working_frame = self.frame.copy()

        working_frame = working_frame[working_frame.date >= working_frame.burnin]
        exposure_group = 'cohortid' if not self.is_bootstrap else 'pseudo_cid'

        if group:
            exposure = working_frame.\
                groupby(group).\
                apply(lambda x : self.getExposure(working_frame=x))
        else:
            exposure = working_frame[exposure_group].unique().size

        return exposure
    def fit(self, group=None):
        """
        calculate force of infection across dataset given groups
        """

        durations = self.getDurations(group=group)
        events = self.getInfections(group=group)
        exposure = self.getExposure(group=group)

        foi = events / (exposure * durations)

        if group:
            grouped_results = np.vstack([
                i.values for i in [events, durations, exposure, foi]
            ]).T
            foi = pd.DataFrame(
                grouped_results,
                columns = ['events', 'durations', 'exposure', 'FOI'],
                index = foi.index
                )

        else:
            foi = pd.DataFrame(
                np.array([events, durations, exposure, foi]).reshape(1,-1),
                columns = ['events', 'durations', 'exposure', 'FOI']
            )

        return foi
class BootstrapCID:
    """
    Perform bootstrapping on a dataframe by cohortid
    requires : `cohortid` as column
    """
    def __init__(self, dataframe, grouping='cohortid', seed=None):
        self.frame = dataframe.set_index(grouping)
        self.cid = self.frame.index.unique()
        self.grouping = grouping

        if seed:
            np.random.seed(seed)
    def sampleCID(self, size = 0, replace=True):
        """
        Random choice of COI in set
        """
        if size == 0:
            size = self.cid.size
        return np.random.choice(self.cid, size, replace=replace)
    def labelPseudo(self, bootstrap):
        """
        Label multiple occurences of a Cohortid with a pseudonym
        """
        group = ['cohortid', 'date']

        pseudo_label = bootstrap.groupby(group).apply(
            lambda x : np.arange(x.shape[0])
        )

        bootstrap.sort_values(group, inplace=True)

        bootstrap['pseudo_cid'] = bootstrap.cohortid.astype(str) + '_' + np.hstack(pseudo_label).astype(str)


        return bootstrap
    def getSample(self, size=0, pseudo_id=True):
        """
        Select cohortids found in sampling and return
        """

        bootstrap = self.frame.loc[self.sampleCID()].\
            reset_index()
        if pseudo_id:
            return self.labelPseudo(bootstrap)

        return bootstrap
    def getIter(self, num_iter=100):
        """
        Iterable generator for bootstraps
        """
        for _ in tqdm(np.arange(num_iter)):
            yield self.getSample()


def dev_infectionLabeler():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta, by_individual=True)
    infection_labels = il.LabelInfections()

    # bootstrapped_meta = BootstrapCID(meta, seed=42).getSample()

    # il = InfectionLabeler(sdo, bootstrapped_meta, qpcr_threshold=0, by_individual=True)
    # infection_labels = il.LabelInfections()

    #
    # no_geno = il.meta[~il.meta.cohortid.isin(infection_labels.cohortid)]
    # no_geno_past_burnin = no_geno[no_geno.date > no_geno.burnin]
    # no_geno_past_burnin[no_geno_past_burnin.qpcr > 0][['cohortid', 'date']].groupby('cohortid').apply(lambda x : x.date.count())

def dev_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta, qpcr_threshold=0)
    infection_labels = il.LabelInfections()
    # infection_labels.to_csv('temp/labels.tab', sep="\t", index=False)

    labels = pd.read_csv('temp/labels.tab', sep="\t")
    foi = FOI(labels, meta)

    grouped = foi.fit(group = ['agecat', 'gender'])
    full = foi.fit(group=None)
def dev_BootstrapLabels():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)
    labels = pd.read_csv('temp/labels.tab', sep="\t")

    bl = BootstrapCID(meta, seed=42)


    total = []
    for b_meta in bl.getIter(100):
        b_labels = InfectionLabeler(sdo, b_meta, by_individual=True).LabelInfections()
        foi = FOI(b_labels, b_meta)
        full = foi.fit(group=None)
        total.append(full)
    bs_foi = pd.concat(total)


    labels = InfectionLabeler(sdo, meta, by_individual=True).LabelInfections()
    true_foi = FOI(labels, meta)
    true_fit = true_foi.fit()

    sns.distplot(bs_foi.FOI)
    plt.axvline(true_foi.FOI[0])

def worker_foi(sdo, meta):
    labels = InfectionLabeler(sdo, meta, by_individual=True).LabelInfections()
    foi = FOI(labels, meta)
    full = foi.fit(group=None)
    return full

def multiprocess_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    bl = BootstrapCID(meta, seed=42)

    p = Pool(processes=7)
    results = p.starmap(
        worker_foi, [(sdo, bl.getSample()) for _ in range(100)]
    )
    results = pd.DataFrame(results)
    print(results)

    pd.DataFrame(np.vstack(results))[3].hist()

    labels = InfectionLabeler(sdo, meta, by_individual=True).LabelInfections()
    foi = FOI(labels, meta)
    foi.fit()

if __name__ == '__main__':
    # dev_infectionLabeler()
    # dev_FOI()
    # dev_BootstrapLabels()
    multiprocess_FOI()
