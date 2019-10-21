#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from pkgpr2.pkgpr2 import InfectionLabeler
from pkgpr2.pkgpr2 import FOI
from pkgpr2.pkgpr2 import ExponentialDecay
from pkgpr2.pkgpr2 import OldWaning
from pkgpr2.pkgpr2 import FractionOldNew
from pkgpr2.pkgpr2 import OldNewSurival


def load_inputs():
    sdo = pd.read_csv(
        '../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv(
        '../prism2/stata/full_meta_6mo.tab', sep="\t",
        low_memory=False)
    return sdo, meta


def load_labels(clone=True):
    fn = "labels.{}.tab"
    if clone:
        fn = fn.format('clone')
    else:
        fn = fn.format('ifx')

    labels = pd.read_csv(fn, sep="\t")
    labels.date = labels.date.astype('datetime64')
    labels.enrolldate = labels.enrolldate.astype('datetime64')
    labels.burnin = labels.burnin.astype('datetime64')

    return labels


def dev_infectionLabeler():
    sdo, meta = load_inputs()

    il = InfectionLabeler(
        sdo, meta, haplodrop=False,
        qpcr_threshold=0, burnin=2
        )

    # labels_a = il.LabelInfections(by_clone=True)
    labels_b = il.LabelInfections(by_clone=False)

    labels_b.to_csv('labels.ifx.tab', sep="\t", index=False)


def dev_FOI():
    sdo, meta = load_inputs()
    labels = InfectionLabeler(sdo, meta)

    foi = FOI(labels, meta, burnin=2)
    print(foi)


def DecayByGroup(infections, n_iter=200, group=['gender'], label=None):
    ed_classes = []
    estimated_values = []
    bootstrapped_values = []
    indices = []
    for index, frame in infections.groupby(group):
        ed = ExponentialDecay(frame)
        l, bsl = ed.fit(bootstrap=True, n_iter=n_iter)
        print(index)
        print(ed.num_classes / ed.num_classes.sum())

        indices.append(index)
        ed_classes.append(ed)
        estimated_values.append(l)
        bootstrapped_values.append(bsl)

    for i, _ in enumerate(ed_classes):
        sns.distplot(1 / bootstrapped_values[i], bins=30)
        plt.axvline(
            1 / estimated_values[i],
            label=indices[i],
            color=sns.color_palette()[i],
            linestyle=':', lw=5
            )
        plt.legend(labels=indices)

    if label:
        plt.savefig('../plots/durations/{}.png'.format(label))
    plt.show()


def dev_Survival():
    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    fon = FractionOldNew(
        infections=labels, meta=meta, burnin=2, bootstrap=False, n_iter=5)
    fon.fit()
    fon.plot()

    ons = OldNewSurival(
        infections=labels, meta=meta, burnin=2, bootstrap=False, n_iter=5)
    ons.fit()
    ons.plot()

    w = OldWaning(
        infections=labels, meta=meta, burnin=2, bootstrap=True, n_iter=5)
    w.fit()
    w.plot()


def durations_by_cat(sub_labels, label='label'):
    e = ExponentialDecay(sub_labels, seed=42)
    l1, l2, durations = e.GetInfectionDurations(sub_labels)

    frame = pd.DataFrame(durations, columns=['classification', 'duration'])
    frame = frame[frame.classification != 0]

    d = {
        1: 'uncensored',
        2: 'c_left',
        3: 'c_right',
        4: 'c_both'
        }

    frame['classification'] = [d[i] for i in frame.classification]
    frame['label'] = label

    return frame


def plot_durations_by_baseline(labels):
    """
    Plot durations by baseline condition for a given label set
    """
    baseline = durations_by_cat(
        labels[labels.active_baseline_infection],
        label='baseline'
        )

    newifx = durations_by_cat(
        labels[~labels.active_baseline_infection],
        label='newifx'
        )
    durations = pd.concat([baseline, newifx])

    print(
        durations.groupby(['classification', 'label']).apply(lambda x: x.size)
        )

    durations.sort_values('classification', inplace=True, ascending=False)

    sns.violinplot(
        data=durations,
        x='classification',
        y='duration',
        cut=0, hue='label'
        )

    plt.show()


def dev_Durations():
    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    plot_durations_by_baseline(labels)

    # e = ExponentialDecay(labels, seed=42)
    # l1, l2, durations = e.GetInfectionDurations(labels)

    # sys.exit()
    DecayByGroup(
        labels, n_iter=500,
        group=['active_baseline_infection']
        )


if __name__ == '__main__':
    # dev_infectionLabeler()
    # dev_Survival()
    dev_Durations()
    # dev_FOI()
    # dev_BootstrapLabels()
    # multiprocess_FOI()
    pass
