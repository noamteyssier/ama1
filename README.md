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
pip install numpy pandas scipy --user
```
```R
# in R
install.packages(c('tidyverse','readstata13','tidygraph','ggraph'))
```

#### fixMeta
- a collection of intermediary steps describing date changes for prism2
- steps describing changes described in processing notes
