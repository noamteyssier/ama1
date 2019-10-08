#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

from pkgpr2.pkgpr2 import InfectionLabeler, FOI, ExponentialDecay
from pkgpr2.pkgpr2 import OldWaning, FractionOldNew, OldNewSurival




def dev_infectionLabeler():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta,
        by_infection_event=False, qpcr_threshold=1, drop_missing=True,
        burnin=2, haplodrops=False, skip_threshold=3)
    labels = il.LabelInfections()
    print(labels)


def dev_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)
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
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    il = InfectionLabeler(sdo, meta,
        by_infection_event=False, qpcr_threshold=0.1,
        burnin=2, haplodrops=False)
    labels = il.LabelInfections()

    fon = FractionOldNew(infections=labels, meta=meta, burnin=2, bootstrap=False, n_iter=5)
    fon.fit()
    fon.plot()

    ons = OldNewSurival(infections=labels, meta=meta, burnin=2, bootstrap=False, n_iter=5)
    ons.fit()
    ons.plot()

    w = OldWaning(infections= labels, meta=meta, burnin=2, bootstrap=True, n_iter=5)
    w.fit()
    w.plot()

    labels.date = labels.date.astype('datetime64')
    e = ExponentialDecay(infections = labels)
    e.fit(bootstrap=True)
    e.plot()

    DecayByGroup(labels, group='agecat')

def worker_foi(sdo, meta, group):
    labels = InfectionLabeler(sdo, meta, by_infection_event=True, impute_missing=True).LabelInfections()
    foi = FOI(labels, meta)
    full = foi.fit(group=group)
    return full


def multiprocess_FOI():
    sdo = pd.read_csv('../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv('../prism2/stata/full_meta_grant_version.tab', sep="\t", low_memory=False)

    bl = BootstrapCID(meta, seed=42)

    p = Pool(processes=7)

    group = ['gender']
    bootstrapped_meta = [[sdo, bl.getSample(), group] for _ in tqdm(range(100))]
    results = p.starmap(worker_foi, bootstrapped_meta)

    bootstrapped_foi = pd.concat(results)

    labels = InfectionLabeler(sdo, meta, by_infection_event=True, impute_missing=True).LabelInfections()
    foi = FOI(labels, meta)
    true_foi = foi.fit(group=group)

    for g, sub in bootstrapped_foi.groupby(group):
        sns.distplot(sub.FOI.values, label=g)
    [plt.axvline(val) for val in true_foi.FOI.values]
    plt.legend()


if __name__ == '__main__':
    dev_infectionLabeler()
    # dev_Survival()
    # dev_FOI()
    # dev_BootstrapLabels()
    # multiprocess_FOI()
    pass
