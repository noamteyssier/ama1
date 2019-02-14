#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import sys
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
        sdu = SeekDeepUtils()
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
            plt.show()
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
            plt.show()
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
            plt.show()
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
    def __init__(self):
        self.sdo = pd.DataFrame()
        pd.set_option('mode.chained_assignment', None) # remove settingwithcopywarning
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
        """generate a longform timeline for a given cohortid"""
        long_cid = self.meta[
            self.meta.cohortid == cohortid
        ].merge(
            self.sdo[['date', 'cohortid', 'h_popUID', 'c_AveragedFrac']],
            how = 'left'
        )

        long_cid['h_fraction'] = long_cid.apply(
            lambda x : x.c_AveragedFrac * x.qpcr, axis = 1)

        wide_cid = long_cid[['date', 'cohortid', 'h_popUID', 'h_fraction']].pivot(
            index='h_popUID',
            columns = 'date',
            values = 'h_fraction')

        wide_cid = wide_cid[~wide_cid.index.isna()]
        if boolArray:
            return ~wide_cid.isna()
        else:
            return wide_cid
    def __all_timelines__(self):
        """returns a dictionary of all timelines indexed by cohortid"""
        return {cid : self.__generate_timeline__(cid, boolArray=True) for cid in self.sdo.cohortid.unique()}
    def __infection_labeller__(self, row, allowedSkips):
        """label infections as true or false"""
        # first visit is always false
        if row.visit_num == 1:
            return False
        # first infection occurs at a timepoint past the allowed skips
        elif row.skips > allowedSkips :
            return True
        # first infection before the skip threshold but there is a gap between first visit and first infection
        elif row.skips > 0 and row.visit_num <= allowedSkips:
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
        # add the position of the initial infection as the first skip
        row.skips = np.insert(row.skips, 0, row.i_events[0])

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
    def __prepare_meta__(self, meta):
        """prepare meta data for usage in timeline generation"""
        self.meta = meta[['date', 'cohortid', 'qpcr']]
        self.meta['date'] = self.meta['date'].astype('str')
        self.meta['cohortid'] = self.meta['cohortid'].astype('str')
        self.meta.sort_values(by='date', inplace=True)
        self.meta = self.meta[~self.meta.qpcr.isna()]
        return self.meta
    def __prepare_sdo__(self, sdo, controls=False):
        """prepare seekdeep output dataframe for internal usage"""
        # keep only patient samples and normalize dataframe
        if controls == False:
            self.sdo = sdo[~sdo.s_Sample.str.contains('ctrl|neg')]
            self.sdo = self.fix_filtered_SDO(self.sdo)
        else:
            self.sdo = sdo

        # split cid and date
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x),
            axis = 1, result_type = 'expand')

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

            # find infection events and skips between them
            timeline[['i_events', 'skips']] = timeline.apply(
                lambda x : self.__calculate_skips__(x, diagnose=True),
                axis = 1, result_type = 'expand')

            # for haplotype row create a dataframe of skips and dates
            for _, row in timeline.iterrows():
                skip_dataframe.append(self.__arrange_skips__(row, cid))

        self.skip_df = pd.concat(skip_dataframe)
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
    def __foi_exposure__(self, meta):
        """calculate exposure for a given sdo dataframe"""
        scalar_exposure = meta['cohortid'].\
            drop_duplicates().\
            size
        return scalar_exposure
    def __foi_exposure_duration__(self, sdo):
        """calculate duration in years for a given sdo dataframe"""
        sdo.date = pd.to_datetime(sdo.date, format = '%Y/%m/%d')
        diff = sdo.date.max() - sdo.date.min()
        return diff.days / 365.25
    def __foi_collapse_infection_by_person__(self):
        """collapse infections of a person_datetime to a single value"""
        personal_infection = self.sdo.\
            groupby('s_Sample').\
            agg({'infection_event' : 'max'})
        self.sdo['person_infection'] = self.sdo.apply(
            lambda x : personal_infection.infection_event[x.s_Sample], axis = 1)
        self.sdo = self.sdo[
            ['s_Sample', 'date', 'cohortid', 'visit_num', 'person_infection']
            ].\
            drop_duplicates()
    def __foi_method_all__(self):
        """calculate force of infection over the entire dataset"""

        new_infections = self.sdo.infection_event.sum()
        duration = self.__foi_exposure_duration__(self.sdo)
        exposure = self.__foi_exposure__(self.meta)
        foi = new_infections / (duration * exposure)

        # return params as pandas dataframe
        params = pd.DataFrame({
            'infection_event' : [new_infections],
            'duration' : [duration],
            'exposure' : [exposure],
            'force_of_infection' : [foi]})

        return params
    def __foi_method_month__(self):
        """calculate force of infection by month and by clone"""
        self.sdo['ym'] = self.sdo.date.dt.to_period('M')
        monthly_infections = self.sdo.\
            groupby('ym').agg({
                'infection_event' : 'sum',
                'date' : lambda x : (max(x) - min(x)).days / 365.25
                }).\
            reset_index()
        monthly_infections['exposure'] = self.__foi_exposure__(self.meta)
        monthly_infections['foi'] = monthly_infections.apply(
            lambda x : x.infection_event / (x.date * x.exposure),
            axis=1)
        return monthly_infections
    def __foi_method_agecat__(self):
        sdo = self.__prepare_sdo__(sdo)
        self.__prepare_meta__(meta)

        print(sdo)
    def __foi_method_individual__(self):
        """calculate foi by person-infection and not clone infection"""
        self.__foi_collapse_infection_by_person__()
        self.sdo['ym'] = self.sdo.date.dt.to_period('M')

        monthly_infections = self.sdo.\
            groupby('ym').agg({
                'person_infection' : 'sum',
                'date' : lambda x : (max(x) - min(x)).days / 365.25
                }).\
            reset_index()
        monthly_infections['exposure'] = self.__foi_exposure__(self.meta)
        monthly_infections['foi'] = monthly_infections.apply(
            lambda x : x.person_infection / (x.date * x.exposure),
            axis=1)
        return monthly_infections
    def __foi_method_person__(self):
        print(self.sdo)
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
        self.__prepare_sdo__(sdo, controls)

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
    def Haplotype_Skips(self, sdo, meta, controls=False):
        """returns an array of the skips found per cid~haplotype across dates"""
        self.__prepare_sdo__(sdo, controls)
        self.__prepare_meta__(meta)
        self.__prepare_skips__()
        return self.skip_df

        # timelines = [self.__generate_timeline__(c, boolArray=True) for c in self.sdo.cohortid.unique()]
        # vals = pd.concat([t.apply(
        #     lambda x : self.__calculate_skips__(x), axis = 1) for t in timelines])
        #
        # vals = np.hstack(vals.values)
        # return vals
    def Duration_of_Infection(self, sdo, meta, controls=False, allowedSkips = 3, default=15):
        """calculates duration of infections for each cohortid ~ h_popUID"""
        self.__prepare_sdo__(sdo)
        self.__prepare_meta__(meta)

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
    def New_Infections(self, sdo, meta, controls=False, allowedSkips = 3):
        """calculates number of new infections for each haplotype in each cohortid with allowed skips"""
        self.__prepare_sdo__(sdo, controls)
        self.__prepare_meta__(meta)
        self.__prepare_skips__()
        self.__label_new_infections__(allowedSkips)

        return self.skip_df

        # # generate timelines for each cohortid~h_popUID
        # timelines = {c : self.__generate_timeline__(c, boolArray=True) for c in self.sdo.cohortid.unique() }#if c == '3149'}
        #
        # # lists to grow for placement into dataframe
        # cids = []
        # hids = []
        # new_ifx = []
        #
        # # iterate through cid~timelines
        # for cohortid, timeline in timelines.items():
        #
        #     # cut timelines for indices where infection events are found by haplotype
        #     haplotypes, dates_found = np.array(np.where(timeline>0)) # 2d arr [[haplotypes found] [dates found]]
        #
        #     # iterate through haplotypes and calculate number of new infections
        #     for h in np.unique(haplotypes):
        #
        #         # haplotype specific dates
        #         dates = dates_found[np.where(haplotypes == h)]
        #
        #         # calculated number of new infections for haplotype
        #         new_infections = self.__haplotype_infections__(dates, allowedSkips)
        #
        #         # grow lists
        #         cids.append(cohortid)
        #         hids.append(timeline.index[h])
        #         new_ifx.append(new_infections)
        #
        # # prepare dataframe for printout
        # self.__prepare_new_infections__(cids, hids, new_ifx)
        # return self.new_infections
    def Force_of_Infection(self, sdo, meta, controls=False, foi_method = 'all', allowedSkips = 3, default=15):
        """calculate force of infection for a dataset"""
        self.__prepare_sdo__(sdo)
        self.__prepare_meta__(meta)
        self.__prepare_skips__()
        self.__label_new_infections__(allowedSkips)
        self.sdo = self.sdo.merge(self.skip_df, how='left')
        self.sdo.date = pd.to_datetime(self.sdo.date, format="%Y-%m-%d")

        if foi_method == 'all':
            return self.__foi_method_all__()
        elif foi_method == 'month':
            return self.__foi_method_month__()
        elif foi_method == 'agecat':
            self.__foi_method_agecat__()
        elif foi_method == 'individual':
            return self.__foi_method_individual__()
        elif foi_method == 'person':
            self.__foi_method_person__()
