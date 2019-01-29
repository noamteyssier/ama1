# TODO for AMA1

## Summarise Run
- number of reads per fastq
- number of dropouts post-mapping
- number of chimeras per sample

## Haplotype Filtering
- filter haplotypes with OOSSLF method
  - one off same sample low frequency
- filter haplotypes with LF method
  - low frequency
- compare differences between each method

## Parameter Optimization
- add qpcr and age as linear parameters to estimate
  - update prism2.R to write dataframe with qpcr and age as features
- fit parameters with filtered and unfiltered haplotypes