# ama1

## Collection of Scripts and Data for PFAMA1 experiments
## prism2
A collection of scripts used to analyze the sequencing data from PRISM2 ama1 hemi-nested pcr experiments.


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
