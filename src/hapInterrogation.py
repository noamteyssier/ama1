#!/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse, sys
from ama1 import InfectionLabeler, FOI

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--seekdeep_output', required=False,
        default="../prism2/full_prism2/final_filter.tab",
        help="SeekDeep Output to use as input to functions")
    p.add_argument('-m', '--meta', required=False,
        default= "../prism2/stata/full_meta_6mo_fu.tab",
        help="Cohort Meta information (tsv) to relate cohortids")
    p.add_argument('-l', '--label_infections',
        action='store_true',
        help="Create a dataframe showing the start and end of infection of each cohortid~h_popUID"
        )
    p.add_argument('-f', '--force_of_infection',
        type=str,
        help="Calculate the Force of Infection of the Population (give grouping variables comma separated or write 'none' for no grouping)")
    p.add_argument('-n', '--allowedSkips',
        default = 3,
        type=int,
        help="Number of allowed skips to allow during calculation of durations (default = 3 skips)"
        )
    p.add_argument('-b', '--burnin',
        default=3,
        type=int,
        help="Number of months to consider a patient in burnin period (default = 3 months)"
        )
    p.add_argument('-q', '--qpcr_threshold',
        default=0,
        type=float,
        required=False,
        help="qpcr threshold to consider a sample (set to 0 for no threshold)"
        )
    p.add_argument('--by_infection_event',
        action='store_true',
        required=False,
        help='Collapse infection events by individual'
        )
    p.add_argument('--no_impute',
        action='store_false',
        required=False,
        help='Dont impute missing haplotypes when collapsing by date '
        )
    p.add_argument('--no_aggregation',
        action='store_false',
        required=False,
        help='Dont aggregate infection events following the skip rule when collapsing by date'
        )

    # if no args given print help
    if len(sys.argv) == 1:
        p.print_help(sys.stderr)
        sys.exit(1)

    args = p.parse_args()
    return args
def print_out(df):
    """simple printout for a pandas dataframe to stdout"""
    df.to_csv(sys.stdout, sep="\t", index=False)

def label_infections(sdo, meta, args):
    il = InfectionLabeler(
        sdo, meta,
        qpcr_threshold = args.qpcr_threshold,
        burnin = args.burnin,
        allowedSkips = args.allowedSkips,
        by_infection_event = args.by_infection_event,
        impute_missing = args.no_impute,
        agg_infection_event = args.no_aggregation
    )

    labels = il.LabelInfections()
    labels.to_csv(sys.stdout, sep="\t", index=False)
def foi(sdo, meta, args):
    group = args.force_of_infection.split(',')

    if group[0] == 'none':
        group = None

    else:
        known_groupings = set(meta.columns)
        known_groupings.add('year_month')

        known_groupings = np.array([i for i in known_groupings])

        given_groupings = np.array([g in known_groupings for g in group])
        if np.any(~given_groupings):
            print('Error :')
            print("\nunknown values in given set : \n", group)
            print('\navailable groupings : \n', np.sort(known_groupings))
            print('\nor "none" for no grouping')
            sys.exit()

    il = InfectionLabeler(
        sdo, meta,
        qpcr_threshold = args.qpcr_threshold,
        burnin = args.burnin,
        allowedSkips = args.allowedSkips,
        by_infection_event = args.by_infection_event,
        impute_missing = args.no_impute,
        agg_infection_event = args.no_aggregation
    )

    labels = il.LabelInfections()
    f = FOI(labels, meta, burnin = args.burnin)
    result = f.fit(group = group)
    result.to_csv(sys.stdout, sep="\t", index=True)
def main():
    args = get_args()
    sdo = pd.read_csv(args.seekdeep_output, sep = "\t")
    meta = pd.read_csv(args.meta, sep="\t", low_memory=False)

    if args.label_infections:
        label_infections(sdo, meta, args)
    elif args.force_of_infection:
        foi(sdo, meta, args)

if __name__ == '__main__':
    main()
