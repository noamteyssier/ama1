#!/usr/bin/env python3

import subprocess as sp
import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from ggplot import *
import sys

class HaplotypeSet:
    def __init__(self, sdo_fn, fasta_fn):
        self.sdo_fn = sdo_fn
        self.fasta_fn = fasta_fn

        self.sdo = pd.DataFrame()
        self.dist = pd.DataFrame()
        self.vcf = pd.DataFrame()
        self.filtered_df = pd.DataFrame()

        self.available_filters = {
            'lfh' : self.__filter_lfh__,
            'lfs' :  self.__filter_lfs__,
            'lfhu' : self.__filter_lfhu__,
            'lfsu' : self.__filter_lfsu__,
            'ooslf': self.__filter_ooslf__
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
        self.sdo = pd.read_csv(self.sdo_fn, sep="\t")

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
    def __prepare_haplotype_dataframe(self):
        pass
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
            rename(index=str, columns={'h_frequency' : 's_frequency'}).\
            merge(self.vcf, on = 'POS', how='inner')
    def __run_filter__(self):
        """call appropriate filter using dictionary"""
        self.available_filters[self.filter_method]()
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
        pass
    def __filter_lfsu__(self):
        """filter haplotypes with low frequency snps AND unknown snps"""
        pass
    def __filter_ooslf__(self):
        """filter haplotypes with conditions : one off haplotype in sample AND low frequency"""
        pass
    def __print_df__(self):
        """write dataframe as TSV"""
        self.filtered_df.to_csv(self.output_filename, sep = "\t", index = False)
    def filter(self, filter_method, frequency, output_filename):
        """error checking and call appropriate filter method"""
        self.filter_method = filter_method
        self.frequency = frequency
        self.output_filename = output_filename

        self.__check_filter__()
        self.__check_frequency__()
        self.__check_output_filename__()

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
        help="type of filter method [lfh, lfs, lfhu, lfsu, ooslf]")
    p.add_argument('-f', '--frequency', required=False,
        help="frequency to use as threshold as float (default = 0.05)")
    p.add_argument('-o', '--output_filename', required=False,
        help="tab delim file of haplotypes passing filter (default = stdout)")
    p.add_argument('-c', '--snp_occurrence', required=False,
        help="number of occurrences a snp must have to be considered (default = 0)")
    args = p.parse_args()

    return args

def main():
    args = get_args()
    h = HaplotypeSet(args.seekdeep_output, args.fasta)
    h.filter(args.filter_method, args.frequency, args.output_filename)



if __name__ == '__main__':
    # plt.show()
    main()
