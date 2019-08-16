#!/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse
from seekdeep_modules import SeekDeepUtils
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--seekdeep_output', required=False,
        default="../prism2/full_prism2/final_filter.tab",
        help="SeekDeep Output to use as input to functions")
    p.add_argument('-m', '--meta', required=False,
        default= "../prism2/stata/filtered_visits.dta",
        help="Cohort Meta information (stata13) to relate cohortids")
    p.add_argument('-a', '--allele_frequency', action='store_true',
        help="Calculate allele frequencies of haplotypes in the population")
    p.add_argument('-s', '--haplotype_skips', action='store_true',
        help="Create a vector of the skips found in the population")
    p.add_argument('-g', '--new_infections', action='store_true',
        help='Create a dataframe for the number of new infections of cohortid~h_popUID')
    p.add_argument('-d', '--durations', action='store_true',
        help="Create a dataframe showing the duration of infection of each cohortid~h_popUID")
    p.add_argument('-l', '--old_new_infections', action='store_true',
        help="Create a dataframe showing the start and end of infection of each cohortid~h_popUID")
    p.add_argument('-f', '--force_of_infection', type=str,
        help="Calculate the Force of Infection of the Population [all, all_individual, all_individual_agecat, all_agecat, month, month_individual, month_agecat, month_individual_agecat, cid, cid_individual]")
    p.add_argument('-n', '--num_skips', default = 3, type=int,
        help="Number of allowed skips to allow during calculation of durations (default = 3 skips)")
    p.add_argument('-x', '--default_duration', default=15, type=int,
        help="Default duration rate to use for single event infections (default = 15 days)")
    p.add_argument('-b', '--burnin', default=3, type=int,
        help="Number of months to consider a patient in burnin period (default = 3 months)")
    p.add_argument('-t', '--qpcr_threshold', default=0.1, type=float,
        help="qpcr threshold to consider a sample (set to 0 for no threshold)")
    p.add_argument('-q', '--fail_flag', action='store_false', default=True,
        help="if no haplotypes are recovered and PCR is positive, drop sample (default=True, use flag to deactivate filter)")

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
    s = SeekDeepUtils(
        sdo = sdo,
        meta = meta,
        fail_flag=args.fail_flag,
        qpcr_threshold=args.qpcr_threshold,
        burnin=args.burnin
        )

    # calculate allele frequency
    if args.allele_frequency:
        a_frequency = s.Time_Independent_Allele_Frequency(sdo)
        return print_out(a_frequency)

    # calculate haplotype skips
    elif args.haplotype_skips:
        hapSkips = s.Haplotype_Skips(sdo, meta)
        return print_out(hapSkips)

    # calculate duration of infections
    elif args.durations:
        durations = s.Duration_of_Infection(
            sdo, meta,
            allowedSkips = args.num_skips,
            default=args.default_duration)
        return print_out(durations)

    # calculate start and end of infections
    elif args.old_new_infections:
        onl = s.Old_New_Infection_Labels(
            allowedSkips = args.num_skips,
            default=args.default_duration,
            burnin=args.burnin)
        return print_out(onl)

    elif args.new_infections:
        new_infections = s.New_Infections(allowedSkips=args.num_skips)
        return print_out(new_infections)

    elif args.force_of_infection:
        foi_params = s.Force_of_Infection(
            foi_method=args.force_of_infection,
            allowedSkips=args.num_skips,
            default=args.default_duration,
            burnin=args.burnin)
        return print_out(foi_params)


if __name__ == '__main__':
    main()
