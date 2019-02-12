#!/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse
from seekdeep_modules import SeekDeepUtils
import sys

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--seekdeep_output', required=False,
        default="../prism2/full_prism2/pfama1_sampInfo.tab.txt",
        help="SeekDeep Output to use as input to functions")
    p.add_argument('-m', '--meta', required=False,
        default= "../prism2/stata/allVisits.dta",
        help="Cohort Meta information (stata13) to relate cohortids")
    p.add_argument('-f', '--allele_frequency', action='store_true',
        help="Calculate allele frequencies of haplotypes in the population")
    p.add_argument('-s', '--haplotype_skips', action='store_true',
        help="Create a vector of the skips found in the population")
    p.add_argument('-d', '--durations', action='store_true',
        help="Create a dataframe showing the duration of infection of each cohortid~h_popUID")
    p.add_argument('-n', '--num_skips', default = 3, type=int,
        help="Number of allowed skips to allow during calculation of durations (default = 3 skips)")
    p.add_argument('-x', '--default_duration', default=15, type=int,
        help="Default duration rate to use for single event infections (default = 15 days)")

    # if no args given print help
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    args = p.parse_args()
    return args
def print_out(df):
    """simple printout for a pandas dataframe to stdout"""
    df.to_csv(sys.stdout, sep="\t", index=False)
def main():
    args = get_args()
    sdo = pd.read_csv(args.seekdeep_output, sep = "\t")
    meta = pd.read_stata(args.meta)

    # initialize modules
    s = SeekDeepUtils()

    if args.allele_frequency:
        a_frequency = s.Time_Independent_Allele_Frequency(sdo)
        return print_out(a_frequency)

    if args.haplotype_skips:
        hapSkips = s.Haplotype_Skips(sdo, meta)
        return print(hapSkips)


    if args.durations:
        durations = s.Duration_of_Infection(
            sdo, meta,
            allowedSkips = args.num_skips,
            default=args.default_duration)
        return print_out(durations)

if __name__ == '__main__':
    main()
