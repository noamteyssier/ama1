#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import sys
from ggplot import *

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
        # return self.oossp
        # self.__melted_oossp__()
        if melted:
            return self.__melted_oossp__()
        else:
            return self.oossp
    def PlotOOSSP(self, vlines, hlines, color_type):
        self.__oossp__()

        if not color_type:
            color_type == 'fraction'

        if color_type == 'fraction':
            self.oossp[['c_Fraction', 'c_Ratio']] = self.oossp.\
                apply(lambda x : self.__flag_hvlines__(x, vlines, hlines),
                result_type = 'expand', axis = 1)

            g = ggplot(self.oossp, aes(x = 'h2_fraction', y = 'majorRatio', color = 'c_Fraction')) +\
                geom_point() +\
                geom_vline(x = vlines, linetype = 'dashed', color = 'grey') +\
                geom_hline(y = hlines, linetype = 'dashed', color = 'grey') +\
                theme_bw()
            print(g)
        elif color_type == 'occurence':
            self.__haplotype_occurence__()
            self.oossp['h2_occurence'] = self.oossp.apply(
                lambda x : self.__flag_occurence__(x), axis = 1)

            g = ggplot(self.oossp, aes(x = 'h2_fraction', y = 'majorRatio', color = 'h2_occurence')) +\
                geom_point(size = 40, alpha = 0.8) +\
                geom_vline(x = vlines, linetype = 'dashed', color = 'grey') +\
                geom_hline(y = hlines, linetype = 'dashed', color = 'grey') +\
                theme_bw() +\
                scale_color_manual(values = ['peru', 'red', 'blue', 'green', 'black'])
            print(g)
        elif color_type == 'density':
            self.__load_meta__()
            self.oossp = self.oossp.merge(self.meta[['s_Sample', 'qpcr']])
            self.oossp = self.oossp[self.oossp.qpcr > 0]
            self.oossp['log10_qpcr'] = self.oossp.apply(
                lambda x : self.__flag_qpcr__(x.qpcr), axis = 1)

            g = ggplot(self.oossp, aes(x = 'h2_fraction', y = 'majorRatio', color = 'log10_qpcr')) +\
                geom_point(size = 50, alpha = 0.5) +\
                geom_vline(x = vlines, linetype = 'dashed', color = 'grey') +\
                geom_hline(y = hlines, linetype = 'dashed', color = 'grey') +\
                theme_bw() +\
                scale_color_manual(values = ['peru', 'red', 'blue', 'green', 'black'])
            print(g)
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
    def __split_cid_date__(self, row):
        """convert s_Sample to date and cohortid"""
        a = row.s_Sample.split('-')
        date, cid = '-'.join(a[:3]), a[-1]
        return [date, cid]
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
        # keep only patient samples and normalize dataframe
        if controls == False:
            self.sdo = sdo[~sdo.s_Sample.str.contains('ctrl|neg')]
            self.sdo = self.fix_filtered_SDO(self.sdo)
        else:
            self.sdo = sdo

        # split cid and date
        self.sdo[['date', 'cid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x),
            axis = 1, result_type = 'expand')

        # initialize dataframe to return
        a_freq = self.sdo[['h_popUID']].drop_duplicates()

        # select h_popUID + cid and drop duplicates created by dates
        hapCounts = self.sdo[['h_popUID', 'cid']].\
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
