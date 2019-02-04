# ama1

## Collection of Scripts and Data for PFAMA1 experiments
## prism2
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
python3 haplotypeFilter.py \
  -i seq_data/filtered_pfama1.fasta \
  -s data/filtered_prism2.tab.txt \
  -m ooslfs \
  -f 0.05
```


## Dependencies
- python - numpy    (pip3 install numpy --user)
- python - pandas   (pip3 install pandas --user)
- R - tidyverse
- R - readstata13
- R - tidygraph
- R - ggraph
- clustal-omega : http://www.clustal.org/omega/#Download
- snp-sites : https://github.com/sanger-pathogens/snp-sites
- snp-dists : https://github.com/tseemann/snp-dists

```bash
# command line
pip3 install numpy pandas --user
```
```R
# in R
install.packages(c('tidyverse','readstata13','tidygraph','ggraph'))
```

#### fixMeta
- a collection of intermediary steps describing date changes for prism2
- steps describing changes described in processing notes
