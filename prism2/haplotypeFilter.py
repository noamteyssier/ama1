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

        self.__generate_resources__()
    def __generate_resources__(self):
        """create : alignment, distance, and vcf"""
        #
        self._aln_fn = self.fasta_fn.replace(".fasta", ".aln")
        self._dists_fn = self._aln_fn.replace(".aln", ".dist")
        self._vcf_fn = self._aln_fn.replace(".aln", ".vcf")

        self.__create_alignment__()
        self.__create_snp_dist__()
        self.__create_snp_vcf__()
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



def get_args():
    """argument handler"""
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--seekdeep_output", help = "tab folder from seekdeep output", required=True)
    p.add_argument("-f", "--fasta", help = "fasta from seekdeep output", required=True)
    args = p.parse_args()
    return args

def main():
    args = get_args()
    h = HaplotypeSet(args.seekdeep_output, args.fasta)



if __name__ == '__main__':
    main()
