#!/usr/bin/env python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pkgpr2.survival as sv
import sys
from multiprocess import Pool

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


def run_survival_function(svf_tuple, n_iter=300):
    base_filename = "../plots/survival/{}.pdf"

    svf, name = svf_tuple
    save = base_filename.format(name)

    sdo, meta = load_inputs()
    labels = load_labels(clone=True)

    s_class = svf(
        infections=labels, meta=meta,
        burnin=2, bootstrap=True, n_iter=n_iter
        )
    s_class.fit()
    s_class.plot(save=save)


def main():
    base_filename = "../plots/survival/{}.pdf"

    survival_functions = [
        (sv.FractionOldNew, 'FractionNew'),
        (sv.OldNewSurvival, 'OldNewMix'),
        (sv.OldWaning, 'OldWaning')
        ]

    p = Pool()
    p.map(run_survival_function, survival_functions)



if __name__ == '__main__':
    main()
