#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pkgpr2.pkgpr2 import InfectionLabeler, FOI, ExponentialDecay
from pkgpr2.pkgpr2 import OldWaning, FractionOldNew, OldNewSurival


def load_inputs():
    sdo = pd.read_csv(
        '../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv(
        '../prism2/stata/full_meta_grant_version.tab', sep="\t",
        low_memory=False)
    return sdo, meta

def dev_infectionLabeler():
    sdo, meta = load_inputs()

    il = InfectionLabeler(
        sdo, meta,
        by_infection_event=False, qpcr_threshold=1, drop_missing=True,
        burnin=2, haplodrops=False, skip_threshold=3)
    labels = il.LabelInfections()
    print(labels)


def dev_FOI():
    sdo, meta = load_inputs()
    labels = InfectionLabeler(sdo, meta)


    foi = FOI(labels, meta, burnin=2)

    full = foi.fit(group = ['year_month'])
    labels.infection_event
    labels[labels.date <= pd.to_datetime('2019-04-01')].infection_event.sum()


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
        sns.distplot(1 /bootstrapped_values[i], bins=30)
        plt.axvline(1/ estimated_values[i], label=indices[i], color=sns.color_palette()[i], linestyle=':', lw=5)
        plt.legend(labels = indices)

    if label:
        plt.savefig('../plots/durations/{}.png'.format(label))
    plt.show()


def dev_Survival():
    sdo, meta = load_inputs()

    il = InfectionLabeler(
        sdo, meta,
        by_infection_event=False, qpcr_threshold=0.1,
        burnin=2, haplodrops=False)
    labels = il.LabelInfections()

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

    e = ExponentialDecay(infections=labels)
    e.fit(bootstrap=True)
    e.plot()

    DecayByGroup(labels, group='agecat')


if __name__ == '__main__':
    dev_infectionLabeler()
    # dev_Survival()
    # dev_FOI()
    # dev_BootstrapLabels()
    # multiprocess_FOI()
    pass
