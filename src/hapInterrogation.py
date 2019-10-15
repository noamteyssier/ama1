#!/usr/bin/env python3


import pandas as pd
import numpy as np
import argparse
import sys
import seaborn as sns
import matplotlib.pyplot as plt

from pkgpr2.pkgpr2 import InfectionLabeler, FOI, ExponentialDecay
from pkgpr2.pkgpr2 import OldWaning, FractionOldNew, OldNewSurival


def get_args():
    p = argparse.ArgumentParser()

    # seekdeep output
    p.add_argument(
        '-i', '--seekdeep_output', required=False,
        default="../prism2/full_prism2/final_filter.tab",
        help="SeekDeep Output to use as input to functions"
        )

    # meta
    p.add_argument(
        '-m', '--meta', required=False,
        default="../prism2/stata/full_meta_6mo_fu.tab",
        help="Cohort Meta information (tsv) to relate cohortids")

    # label flag
    p.add_argument(
        '-l', '--label_infections',
        action='store_true',
        help=(
            "Create a dataframe showing the start and end of"
            "infection of each cohortid~h_popUID"
            )
        )

    # foi flag
    p.add_argument(
        '-f', '--force_of_infection',
        type=str,
        help=(
            "Calculate the Force of Infection of the Population"
            "(give grouping variables comma separated "
            "or write 'none' for no grouping)"
            )
        )

    # durations flag
    p.add_argument(
        '-d', '--durations',
        type=str,
        help=(
            "Estimate duration of infections using an exponential decay model "
            "(give grouping variables comma separated"
            "or write 'none' for no grouping)")
        )

    # survival flag
    p.add_argument(
        '-s', '--survival',
        type=str,
        help=(
            'Plot survival of old/new haplotypes '
            '(options : fraction_oldnew, survival_oldnewmix, waning_old)'
            )
        )

    # skip threshold
    p.add_argument(
        '-n', '--skip_threshold',
        default=3,
        type=int,
        help=(
            "Number of allowed skips to allow during calculation of durations "
            "(default = 3 skips)"
            )
        )

    # burnin
    p.add_argument(
        '-b', '--burnin',
        default=2,
        type=int,
        help=(
            "Number of months to consider a patient in burnin period "
            "(default = 3 months)"
            )
        )

    # qpcr threshold
    p.add_argument(
        '-q', '--qpcr_threshold',
        default=0,
        type=float,
        required=False,
        help="qpcr threshold to consider a sample (set to 0 for no threshold)"
        )

    # by infection event
    p.add_argument(
        '--by_infection_event',
        action='store_false',
        required=False,
        help='Aggregate labels by infection events'
        )

    # no impute
    p.add_argument(
        '--no_impute',
        action='store_false',
        required=False,
        help='Dont impute missing haplotypes when collapsing by date '
        )

    # no aggregation
    p.add_argument(
        '--no_aggregation',
        action='store_false',
        required=False,
        help=(
            'Dont aggregate infection events following the skip '
            'rule when collapsing by date'
            )
        )

    # no drop missing
    p.add_argument(
        "--no_drop_missing",
        action='store_false',
        help=(
            "Count missing genotyping but qpcr positive"
            "visits in skip calculation"
            )
        )

    # number of bootstraps
    p.add_argument(
        "--num_bootstraps",
        default=200,
        type=int,
        help='number of times to bootstrap'
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
        qpcr_threshold=args.qpcr_threshold,
        burnin=args.burnin,
        skip_threshold=args.skip_threshold,
        by_infection_event=args.by_infection_event,
        impute_missing=args.no_impute,
        agg_infection_event=args.no_aggregation
    )

    labels = il.LabelInfections(by_clone=args.by_infection_event)
    return labels


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

    labels = label_infections(sdo, meta, args)
    f = FOI(labels, meta, burnin=args.burnin)
    result = f.fit(group=group)
    result.to_csv(sys.stdout, sep="\t", index=True)


def DecayByGroup(infections, n_iter=100, group=['gender'], label=None):
    ed_classes = []
    estimated_values = []
    bootstrapped_values = []
    indices = []
    for index, frame in infections.groupby(group):
        ed = ExponentialDecay(frame)
        l, bsl = ed.fit(bootstrap=True, n_iter=n_iter)

        indices.append(index)
        ed_classes.append(ed)
        estimated_values.append(l)
        bootstrapped_values.append(bsl)

    for i, _ in enumerate(ed_classes):
        sns.distplot(1 / bootstrapped_values[i], bins=30)
        plt.axvline(
            1 / estimated_values[i], label=indices[i],
            color=sns.color_palette()[i],
            linestyle=':', lw=5
            )
        plt.legend(labels=indices)

    if label:
        plt.savefig('../plots/durations/{}.png'.format(label))
    plt.show()


def durations(sdo, meta, args):
    group = args.durations.split(',')

    if group[0] == 'none':
        group = None
    else:
        known_groupings = set(meta.columns)

        known_groupings = np.array([i for i in known_groupings])

        given_groupings = np.array([g in known_groupings for g in group])
        if np.any(~given_groupings):
            print('Error :')
            print("\nunknown values in given set : \n", group)
            print('\navailable groupings : \n', np.sort(known_groupings))
            print('\nor "none" for no grouping')
            sys.exit()

    labels = label_infections(sdo, meta, args)

    if group:
        DecayByGroup(labels, n_iter=args.num_bootstraps, group=group)
    else:
        e = ExponentialDecay(labels)
        e.fit(bootstrap=True, n_iter=args.num_bootstraps)
        e.plot()


def survival(sdo, meta, args):

    known_methods = ['fraction_oldnew', 'survival_oldnewmix', 'waning_old']
    if args.survival not in known_methods:
        sys.exit('Error : choose a method from :', known_methods)

    labels = label_infections(sdo, meta, args)

    if known_methods.index(args.survival) == 0:
        fon = FractionOldNew(
            infections=labels, meta=meta, burnin=args.burnin,
            bootstrap=True, n_iter=args.num_bootstraps
            )
        fon.fit()
        fon.plot()

    elif known_methods.index(args.survival) == 1:
        ons = OldNewSurival(
            infections=labels, meta=meta, burnin=args.burnin,
            bootstrap=True, n_iter=args.num_bootstraps
            )
        ons.fit()
        ons.plot()
    else:
        w = OldWaning(
            infections=labels, meta=meta, burnin=args.burnin,
            bootstrap=True, n_iter=args.num_bootstraps
            )
        w.fit()
        w.plot()


def main():
    args = get_args()
    sdo = pd.read_csv(args.seekdeep_output, sep="\t")
    meta = pd.read_csv(args.meta, sep="\t", low_memory=False)

    if args.label_infections:
        labels = label_infections(sdo, meta, args)
        labels.to_csv(sys.stdout, sep="\t", index=True)
    elif args.force_of_infection:
        foi(sdo, meta, args)
    elif args.durations:
        durations(sdo, meta, args)
    elif args.survival:
        survival(sdo, meta, args)


if __name__ == '__main__':
    main()
