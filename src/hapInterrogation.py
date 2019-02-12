#!/usr/bin/env python3


import pandas as pd
import numpy as np
from seekdeep_modules import SeekDeepUtils

def main():
    sdo_fn = "../prism2/full_prism2/pfama1_sampInfo.tab.txt"
    meta_fn = "../prism2/stata/allVisits.dta"
    sdo = pd.read_csv(sdo_fn, sep = "\t")
    meta = pd.read_stata(meta_fn)

    s = SeekDeepUtils()
    # a_frequency = s.Time_Independent_Allele_Frequency(sdo)
    # hapSkips = s.Haplotype_Skips(sdo, meta)
    s.Duration_of_Infection(sdo, meta, allowedSkips=2)

if __name__ == '__main__':
    main()
