#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
from ggplot import *
import sys

class HaplotypeUtils:
    def __init__(self, dist, vcf, sdo, meta):
        self.dist_fn = dist
        self.vcf_fn = vcf
        self.sdo_fn = sdo
        self.meta_fn = meta

        self.dist = None # pd.DataFrame()
        self.vcf = None # pd.DataFrame()
        self.sdo = None # pd.DataFrame()
        self.meta = None # pd.DataFrame()

        self.melted_dist = None # pd.DataFrame()
        self.one_off = None # pd.DataFrame()
        self.same_sample_pairs = None # pd.DataFrame()
        self.oossp = None # pd.DataFrame()
        self.melted_oossp = None # pd.DataFrame()

        self.__load_dist__()
        self.__load_vcf__()
        self.__load_sdo__()
        self.__load_meta__()
    def __load_dist__(self):
        """read in filename for snp distances into dataframe"""
        self.dist = pd.read_csv(self.dist_fn, sep="\t").\
            rename(columns = {'snp-dists 0.6.3' : 'h_popUID'})
    def __load_vcf__(self):
        """read in filename variant calls into dataframe"""
        self.vcf = pd.read_csv(self.vcf_fn, sep="\t", skiprows=3).\
            drop(columns = ["#CHROM", "ID", "REF", "ALT", "QUAL", "FILTER","INFO","FORMAT"])
    def __load_sdo__(self):
        """read in filename seekdeep output into dataframe"""
        self.sdo = pd.read_csv(self.sdo_fn, sep="\t")
        self.sdo[['date', 'cohortid']] = self.sdo.apply(
            lambda x : self.__split_cid_date__(x), axis = 1, result_type = 'expand')
    def __load_meta__(self):
        """parses statabase13 for relevant columns"""
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
    def PlotOOSSP(self, vlines, hlines, color_type = 'fraction'):
        self.__oossp__()
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


def main():
    fasta = '/home/noam/bin/ama1/prism2/full_prism2/pfama1.fasta'
    dist = '/home/noam/bin/ama1/prism2/full_prism2/pfama1.dist'
    vcf = '/home/noam/bin/ama1/prism2/full_prism2/pfama1.vcf'
    sdo = '/home/noam/bin/ama1/prism2/full_prism2/pfama1_sampInfo.tab.txt'
    meta = '/home/noam/bin/ama1/prism2/stata/allVisits.dta'

    h = HaplotypeUtils(dist, vcf, sdo, meta)
    h.PlotOOSSP(
        vlines = [0.01, 0.02, 0.03],
        hlines = [50, 100],
        color_type = 'fraction'
        )



if __name__ == '__main__':
    main()
