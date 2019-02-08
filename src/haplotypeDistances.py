#!/usr/bin/env python3

from seekdeep_modules import *
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--sdo_fn', default= '../prism2/full_prism2/pfama1_sampInfo.tab.txt',
        help = 'SeekDeep Output dataframe to filter')
    p.add_argument('-d', '--dist_fn', default= '../prism2/full_prism2/pfama1.dist',
        help = 'Distance matrix of snps to use to calculate one-offs (snp-dists output)')
    p.add_argument('-m', '--meta_fn', default= '../prism2/stata/allVisiits.dta',
        help = 'cohort meta data to use for qpcr colouring (only required for plot : density)')
    p.add_argument('-f', '--filter', action='store_true',
        help = 'perform filter and return filtered dataframe (mutually exclusive with -g flag)')
    p.add_argument('-r', '--ratio', default=50, type=float,
        help= 'ratio of majority haplotype over minority to apply filter')
    p.add_argument('-c', '--percent', default=0.01, type=float,
        help = 'percentage to cut minority haplotype under with ratio')
    p.add_argument('-g', '--plot_graph',
        help = 'visualize minority haplotype population with different color schemes [fraction, occurence, density]')
    args = p.parse_args()

    if args.filter and args.plot_graph:
        sys.exit('Error : Please choose to filter OR to visualize')
    if not args.filter and not args.plot_graph:
        sys.exit('Error : Please choose an option to filter OR to visualize')

    return args
def main():
    args = get_args()
    h = HaplotypeUtils(
        dist = args.dist_fn,
        sdo = args.sdo_fn,
        meta = args.meta_fn)

    if args.filter:
        h.FilterOOSSP(ratio = args.ratio, pc = args.percent)
        return 1

    if args.plot_graph:
        h.PlotOOSSP(
            vlines = [0.01, 0.02, 0.03],
            hlines = [50, 100],
            color_type = args.plot_graph)
        return 1



if __name__ == '__main__':
    main()
