#!/usr/bin/env python3

import subprocess as sp
import pandas as pd
import numpy as np
import argparse

class HaplotypeSet:
    def __init__(self, sdo_fn, fasta_fn):
        self.sdo_fn = sdo_fn
        self.fasta_fn = fasta_fn

        self.sdo = pd.DataFrame()
        self.dist = pd.DataFrame()
        self.vcf = pd.DataFrame()

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
    def __check_filter__(self, filter_method):
        """assertion to confirm filter method is supported"""
        assert filter_method in self.available_filters, \
        "\nFilter Method '{0}' not supported \nMethods Supported : {1}".\
        format(filter_method, ' '.join(self.available_filters))
    def filter(self, filter_method):
        self.__check_filter__(filter_method)
        self.available_filters[filter_method]()
    def __filter_lfh__(self):
        """filter haplotypes with low frequency haplotype (population frequency)"""
        pass
    def __filter_lfs__(self):
        """filter haplotypes with low frequency snps"""
        pass
    def __filter_lfhu__(self):
        """filter haplotypes with low frequency population frequency AND unknown snps"""
        pass
    def __filter_lfsu__(self):
        """filter haplotypes with low frequency snps AND unknown snps"""
        pass
    def __filter_ooslf__(self):
        """filter haplotypes with conditions : one off haplotype in sample AND low frequency"""
        pass

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
    args = p.parse_args()

    return args

def main():
    args = get_args()
    h = HaplotypeSet(args.seekdeep_output, args.fasta)
    h.filter(args.filter_method)



if __name__ == '__main__':
    main()
