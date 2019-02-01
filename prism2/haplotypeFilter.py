#!/usr/bin/env python3

import subprocess as sp
import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from ggplot import *
import sys
import itertools

class HaplotypeSet:
    def __init__(self, sdo_fn, fasta_fn):
        self.sdo_fn = sdo_fn
        self.fasta_fn = fasta_fn

        # seekdeep output table
        self.sdo = pd.DataFrame()

        # snp-dists output table
        self.dist = pd.DataFrame()

        # snp-sites vcf output table
        self.vcf = pd.DataFrame()

        # snp-database output table (malariagen)
        self.sdb = pd.DataFrame()

        # processed dataframe after filter is applied
        self.filtered_df = pd.DataFrame()

        # dataframe created of haplotype pairs with a distance of one
        self.OneOff = pd.DataFrame()

        # dataframe created of haplotypes found in the same sample
        self.SameSample = pd.DataFrame()

        # offset for 3D7 genome and AMA1 position
        self.sdb_offset = 1294307

        # list of samples found in self.sdo
        self.samples = []

        # dictionary switch statement
        self.available_filters = {
            'lfh' : self.__filter_lfh__,
            'lfs' :  self.__filter_lfs__,
            'lfhu' : self.__filter_lfhu__,
            'lfsu' : self.__filter_lfsu__,
            'ou' : self.__filter_ou__,
            'ooslfs': self.__filter_ooslfs__
        }

        self.__generate_resources__()
    def __generate_resources__(self):
        """create : alignment, distance, and vcf"""
        self._aln_fn = self.fasta_fn.replace(".fasta", ".aln")
        self._dists_fn = self._aln_fn.replace(".aln", ".dist")
        self._vcf_fn = self._aln_fn.replace(".aln", ".vcf")

        # self.__create_alignment__()
        # self.__create_snp_dist__()
        # self.__create_snp_vcf__()
        self.__create_dataframes__()
    def __create_alignment__(self):
        """align fasta using clustal omega"""
        sp.run(
            "clustalo -i {0} -o {1} -t dna --force".\
                format(self.fasta_fn, self._aln_fn),
            shell=True)
    def __create_snp_dist__(self):
        """generate snp distance matrix using snp-dists"""
        sp.run(
            "snp-dists -q {0} > {1}".\
                format(self._aln_fn, self._dists_fn),
            shell = True
        )
    def __create_snp_vcf__(self):
        """generate snp vcf using snp-sites"""
        sp.run(
            "snp-sites -v -o {1} {0}".\
                format(self._aln_fn, self._vcf_fn),
            shell=True
        )
    def __create_dataframes__(self):
        """create pandas dataframes and tidy"""
        # read in seekdeep output table
        self.sdo = pd.read_csv(self.sdo_fn, sep="\t")
        self.samples = self.sdo.s_Sample.unique().tolist()

        # rename version number to h_popUID
        self.dist = pd.read_csv(self._dists_fn, sep="\t").\
            rename(columns = {"snp-dists 0.6.3" : 'h_popUID'})

        # keep only positions and haplotypes
        self.vcf = pd.read_csv(self._vcf_fn, sep="\t", skiprows=3).\
            drop(columns = ["#CHROM", "ID", "REF", "ALT", "QUAL", "FILTER","INFO","FORMAT"])
    def __check_filter__(self):
        """assertion to confirm filter method is supported"""
        assert self.filter_method in self.available_filters, \
        "\nFilter Method '{0}' not supported \nMethods Supported : {1}".\
        format(self.filter_method, ' '.join(self.available_filters))
    def __check_frequency__(self):
        """assertion to check frequency is a float between 0 and 1"""
        self.frequency = float(self.frequency) if self.frequency != None else 0.05
        assert self.frequency > 0 and self.frequency < 1, \
            "\nFrequency must be a float between 0 and 1 (user provided {})".\
            format(self.frequency)
    def __check_output_filename__(self):
        """assigns default stdout if filename is not given"""
        self.output_filename = self.output_filename if self.output_filename != None else sys.stdout
    def __check_snp_database__(self):
        """confirms that snp database is present for known snp searching filtering methods"""
        if 'u' in self.filter_method:
            assert self.sdb, "\nFiltering Method '{0}' requires a snp database to cross reference position"
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
    def __get_samples__(self):
        """utility function to retrieve sample names"""
        return self.samples
    def __process_vcf__(self):
        """prepare vcf dataframe for downstream processing"""
        # convert from wide to long
        self.vcf = pd.melt(
                self.vcf,
                id_vars=['POS'],
                value_vars=self.vcf.columns[1:],
                var_name="haplotype",
                value_name='snp_bool'
            )

        # filter out non snp~haplotypes
        self.vcf = self.vcf[self.vcf.snp_bool > 0]

        # convert values greater than 1 to one (not sure why snp-sites gives these)
        self.vcf.snp_bool.replace("[0-9]",1, inplace=True, regex=True)

        # separate haplotype frequency from id
        self.vcf[['hid', 'h_frequency']] = self.vcf.\
            apply(
                lambda x : x['haplotype'].split('_f'), axis = 1, result_type='expand'
            )

        # convert frequency to float
        self.vcf['h_frequency'] = self.vcf['h_frequency'].astype('float')

        # calculate snp frequency as mean of haplotype frequency containing it
        self.vcf = self.vcf.\
            groupby('POS', as_index=False).\
            agg({'h_frequency': 'mean'}).\
            rename(columns={'h_frequency' : 's_frequency'}).\
            merge(self.vcf, on = 'POS', how='inner')

        # calculate snp occurrence
        self.vcf = self.vcf.\
            groupby('POS', as_index=False).\
            agg({'h_frequency': 'count'}).\
            rename(columns={'h_frequency' : 's_occurrence'}).\
            merge(self.vcf, on = 'POS', how='inner')
    def __process_snp_database__(self):
        """load in snp database, split snp position, and calculate relative position to amplicons"""
        self.sdb = pd.read_csv(self.sdb, sep="\t")
        self.sdb['relative_snp_position'] = self.sdb.\
            apply(lambda x : int(x['SNP_Id'].split('.')[-1]) - self.sdb_offset, axis = 1)
    def __process_dists__(self):
        """process snp-dists output into long format of hid-hid-steps + make OneOff"""
        # # keep only lower triangular
        # self.dist = self.__NaN_upper_triangular__(self.dist)

        # melt dataframe for hid-hid-steps
        self.dist = pd.melt(
            self.dist,
            id_vars=['h_popUID'],
            value_vars=self.dist.columns[1:],
            var_name="h_popUID2",
            value_name="steps")

        # only keep pairs with a distance
        self.dist = self.dist[self.dist.steps > 0]

        # remove population frequency from hid
        self.dist['h_popUID'] = self.dist.\
            apply(lambda x : x['h_popUID'].split('_f')[0], axis = 1)
        self.dist['h_popUID2'] = self.dist.\
            apply(lambda x : x['h_popUID2'].split('_f')[0], axis = 1)

        # data frame of haplotypes only one off of each other
        self.OneOff = self.dist[self.dist.steps == 1].rename(columns = {'h_popUID' : 'h_popUID1'})
    def __find_known_snps__(self):
        """identify snps with known positions and return haplotypes snps are found in"""
        self.known_snp_haplotypes = self.vcf[
            self.vcf.POS.isin(self.sdb.relative_snp_position)
            ].hid.unique()
    def __find_same_sample_pairs__(self):
        """identify all haplotypes that appear in the same sample (hid~date)"""
        hap1 = []
        hap2 = []
        for i in self.__get_samples__():
            l = self.sdo[self.sdo.s_Sample == i].h_popUID.tolist()
            if len(l) > 1:
                mat = np.array([[i,j] for (i,j) in itertools.combinations(l, r=2)])
                hap1.append(mat[:,0])
                hap2.append(mat[:,1])
        self.SameSample = pd.DataFrame({
            'h_popUID1' : [item for sublist in hap1 for item in sublist],
            'h_popUID2' : [item for sublist in hap2 for item in sublist]}
            )
    def __run_filter__(self):
        """call appropriate filter using dictionary"""
        self.available_filters[self.filter_method]()
    def __filter_unknown__(self, process_vcf=True):
        """applies filter to keep only haplotypes with known snps"""
        self.__process_snp_database__()
        if process_vcf: self.__process_vcf__()
        self.__find_known_snps__()
        self.filtered_df = self.filtered_df[self.filtered_df.h_popUID.isin(self.known_snp_haplotypes)]
    def __filter_lfh__(self):
        """filter haplotypes with low frequency haplotype (population frequency)"""
        self.filtered_df = self.sdo[self.sdo['h_SampFrac'] >= self.frequency]
    def __filter_lfs__(self):
        """filter haplotypes with low frequency snps"""
        self.__process_vcf__()
        passing_haplotypes = self.vcf[self.vcf.s_frequency >= self.frequency].hid.unique().tolist()
        self.filtered_df = self.sdo[self.sdo.h_popUID.isin(passing_haplotypes)]
    def __filter_lfhu__(self):
        """filter haplotypes with low frequency population frequency AND unknown snps"""
        self.__filter_lfh__()
        self.__filter_unknown__()
    def __filter_lfsu__(self):
        """filter haplotypes with low frequency snps AND unknown snps"""
        self.__filter_lfs__()
        self.__filter_unknown__(process_vcf=False)
    def __filter_ou__(self):
        """filter if one occurrence and snp is unknown"""
        self.__process_vcf__()
        to_filter = self.vcf[self.vcf.s_occurrence == 1].hid.unique()
        self.filtered_df = self.sdo[~self.sdo.h_popUID.isin(to_filter)]
    def __filter_ooslfs__(self):
        """filter haplotypes with conditions : one off haplotype in sample AND low frequency snp"""
        self.__process_dists__()
        self.__process_vcf__()
        self.__find_same_sample_pairs__()

        # one off and same sample
        ooss = self.OneOff.\
            merge(self.SameSample, how = 'inner').\
            merge(self.vcf[self.vcf.s_frequency >= self.frequency], how = 'inner', left_on='h_popUID1', right_on='hid').\
            drop_duplicates()

        # apply filter
        self.filtered_df = self.sdo[self.sdo.h_popUID.isin(ooss.h_popUID1.unique())]
    def __print_df__(self):
        """write dataframe as TSV"""
        self.filtered_df.to_csv(self.output_filename, sep = "\t", index = False)
    def filter(self, filter_method, frequency, output_filename, snp_database):
        """error checking and call appropriate filter method"""
        self.filter_method = filter_method
        self.frequency = frequency
        self.output_filename = output_filename
        self.sdb = snp_database

        self.__check_filter__()
        self.__check_frequency__()
        self.__check_output_filename__()
        self.__check_snp_database__()

        self.__run_filter__()
        self.__print_df__()

def get_args():
    """argument handler"""
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--fasta", required=True,
        help="fasta from seekdeep output")
    p.add_argument("-s", "--seekdeep_output", required=True,
        help="tab folder from seekdeep output")
    p.add_argument("-m", '--filter_method', required=True,
        help="type of filter method [lfh, lfs, lfhu, lfsu, ou, ooslfs]")
    p.add_argument('-f', '--frequency', required=False,
        help="frequency to use as threshold as float (default = 0.05)")
    p.add_argument('-o', '--output_filename', required=False,
        help="tab delim file of haplotypes passing filter (default = stdout)")
    p.add_argument('-c', '--snp_occurrence', required=False,
        help="number of occurrences a snp must have to be considered (default = 0)")
    p.add_argument('-d', '--snp_database', required=False,
        help="snp database to use for unknown snp calls (required for LFHU and LFSU filtering)")
    args = p.parse_args()
    return args
def main():
    args = get_args()
    h = HaplotypeSet(args.seekdeep_output, args.fasta)
    h.filter(
        args.filter_method,
        args.frequency,
        args.output_filename,
        args.snp_database
        )



if __name__ == '__main__':
    main()
