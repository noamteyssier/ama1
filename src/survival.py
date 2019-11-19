#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pkgpr2.survival as sv
import sys


def load_inputs():
    sdo = pd.read_csv(
        '../prism2/full_prism2/final_filter.tab', sep="\t")
    meta = pd.read_csv(
        '../prism2/stata/full_meta_oct_6mo.tab', sep="\t",
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


def run_FractionOldNew(save=None):
    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    fon = sv.FractionOldNew(
        infections=labels, meta=meta,
        burnin=2, bootstrap=True, n_iter=200
        )
    fon.fit()
    fon.plot(save=save)


def run_OldNewSurvival(save=None):
    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    ons = sv.OldNewSurival(
        infections=labels, meta=meta,
        burnin=2, bootstrap=True, n_iter=200
        )
    ons.fit()
    ons.plot(save=save)


def run_OldWaning(save=None):
    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    waning = sv.OldWaning(
        infections=labels, meta=meta,
        burnin=2, bootstrap=True, n_iter=200
        )
    waning.fit()
    waning.plot(save=save)


def main():
    base_filename = "../plots/survival/{}.pdf"

    run_FractionOldNew(
        save=base_filename.format('FractionOldNew')
        )

    run_OldNewSurvival(
        save=base_filename.format('OldNewMix')
        )

    run_OldWaning(
        save=base_filename.format('OldWaning')
        )



if __name__ == '__main__':
    main()
