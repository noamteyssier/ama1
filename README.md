# AMA1
A collection of scripts used to analyze the sequencing data from PRISM2 ama1 hemi-nested pcr experiments.

## haplotypeFilter
A script to apply filters on seekdeep output to remove suspicious haplotypes on
attributes defined by the user. Will return a dataframe in the same format as
SeekDeep's output with the suspicious haplotypes removed.

Currently the filters are [lfh, lfs, lfhu, lfsu, ou, ooslfs]
- lfh : low frequency haplotype in population)
- lfs : low frequency snp in population
- lfhu : lfh + unknown snp
- lfsu : lfs + unknown snp
- ou : one snp occurrence and unknown snp
- ooslfs : one off haplotype in sample and low frequency snp

Frequency is given as a float (ex : 0.05 is 5%)

snp_database is a tab delim file pulled from MalariaGen of known snps for ama1

```bash
# example usage
cd src/
./haplotypeFilter.py \
  -i ../prism2/seq_data/filtered_pfama1.fasta \
  -s ../prism2/data/filtered_prism2.tab.txt \
  -m ooslfs \
  -f 0.05
```

## OOSSP_filter
A filter specifically for one-off-same-sample-pairs (OOSSP) with visualizations and filter options.
Flags include :
- i : seek deep output filename (optional and defaults to full_prism2)
- d : fasta distance filename (optional and defaults to full_prism2)
- m : cohort meta filename (optional and defaults to prism2 statabase)
- r : the majority/minority SSP percentage ratio (defaults to 50) (can be visualized with --plot_graph before choosing)
- c : percentage threshold of minority SSP to be used in combination with ratio (defaults to 0.01) (can be visualized with --plot_graph before choosing)
- f : a flag to filter the SeekDeep dataframe and print to stdout in the same format as input
- g : a flag to plot the OOSSP ratio v. percentage correlation plot of minority SSP with variable color schemes [fraction, occurence, density]
  - fraction : colors by percentage of minor haplotype
  - occurence : colors by occurence of the haplotype in the population
  - density : colors by log10 qPCR density of samples haplotypes are found in

###### Note:
Visualization and filtering cannot be done in the same run. Plot first and then filter

```bash
# example usage to visualize
cd src
./OOSSP_filter.py -g fraction

# example usage to filter with defaults
./OOSSP_filter.py -f

# example usage to filter with ratio of 75 and percentage of 0.02
./OOSSP_filter.py -f -r 75 -c 0.02
```



## fitTimeline
a script to fit models to estimate probability of recovery and sensitivity of detection on a dataset over triplets of haplotypes and possible sample collection dates. Uses the triplet model described in Smith-Felger (PMID:10450427).

###### Note:
- takes seekdeep output as input alongside metadata in stata13 format.
- will filter samples with 'neg' or 'ctrl' in sample name
- will filter out dates of non-routine visits unless malaria episode was recorded

```bash
# example usage
cd src/
./fitTimeline.py \
  -i ../prism2/data/filtered_prism2.tab.txt \
  -c ../prism2/stata/allVisits.dta \
  --seed 42
```

## prism2
An R script to generate visualizations of interest to the project. Names to files are hardcoded into the script so if using filtered dataframes you must change the load in files at the top of the file. Generated plots for unfiltered data are available in prism2/plots.

###### Will Generate:
  - run summary statistics
  - read count summaries by categorical density
  - haplotype fractions for control samples
  - snp frequency distributions between known/unknown snps
  - snp frequencies in the poulation
  - malaria event timelines
  - cid~haplotype density timelines
  - haplotype distance network plots

## Dependencies
- python - numpy-1.15.4
- python - pandas-0.24.1
- python - scipy-1.2.0
- R - tidyverse
- R - readstata13
- R - tidygraph
- R - ggraph
- clustal-omega : http://www.clustal.org/omega/#Download
- snp-sites : https://github.com/sanger-pathogens/snp-sites
- snp-dists : https://github.com/tseemann/snp-dists

```bash
# command line
pip install numpy pandas scipy ggplot --user
```
```R
# in R
install.packages(c('tidyverse','readstata13','tidygraph','ggraph'))
```

#### fixMeta
- a collection of intermediary steps describing date changes for prism2
- steps describing changes described in processing notes
