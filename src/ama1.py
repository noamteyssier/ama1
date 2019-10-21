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

    sys.exit()


    labels_a.infection_event.sum()
    labels_b[labels_b.date < pd.to_datetime('2019-04-01')].infection_event.sum()

    labels_a.groupby('cohortid').apply(lambda x : x.infection_event.sum()).reset_index().to_csv(sys.stdout, sep="\t")
    labels_b.groupby('cohortid').apply(lambda x : x.infection_event.sum()).reset_index().to_csv(sys.stdout, sep="\t")


def dev_FOI():
    sdo, meta = load_inputs()
    labels = InfectionLabeler(sdo, meta)


    foi = FOI(labels, meta, burnin=2)

    full = foi.fit(group = ['year_month'])
    labels.infection_event
    labels[labels.date <= pd.to_datetime('2019-04-01')].infection_event.sum()


def DecayByGroup(infections, n_iter=200, group=['gender'], label=None):
    ed_classes = []
    estimated_values = []
    bootstrapped_values = []
    indices = []
    for index, frame in infections.groupby(group):
        ed = ExponentialDecay(frame)
        l, bsl = ed.fit(bootstrap=True, n_iter=n_iter)
        print(index)
        print(ed.num_classes)

        indices.append(index)
        ed_classes.append(ed)
        estimated_values.append(l)
        bootstrapped_values.append(bsl)

    for i, _ in enumerate(ed_classes):
        sns.distplot(1 /bootstrapped_values[i], bins=30)
        plt.axvline(1/ estimated_values[i], label=indices[i], color=sns.color_palette()[i], linestyle=':', lw=5)
        plt.legend(labels = indices)

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


def dev_Durations():
    sdo, meta = load_inputs()
    labels = load_labels(clone=False)


    # e = ExponentialDecay(labels, seed=42)
    # e.fit(bootstrap=True, n_iter=200)
    # e.plot()

    num_visits = labels.groupby(['cohortid', 'h_popUID']).apply(lambda x : x.shape[0] > 1).reset_index()
    labels = labels.merge(num_visits, how='left')
    labels = labels[labels[0]].drop(columns=0)
    # print(labels)
    # sns.distplot(num_visits)
    # plt.show()

    # sys.exit()
    DecayByGroup(
        labels, n_iter=500,
        group=['active_baseline_infection']#, 'agecat']
        )


if __name__ == '__main__':
    # dev_infectionLabeler()
    # dev_Survival()
    dev_Durations()
    # dev_FOI()
    # dev_BootstrapLabels()
    # multiprocess_FOI()
    pass
