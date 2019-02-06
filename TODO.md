# TODO for AMA1

## Summarise Run
- number of chimeras per sample
- list of everything with zero reads

## snp_frequency
- sum instead of mean?

## Haplotype Filtering
- filter haplotypes with OOSSLF method
  - one off same sample low frequency
- filter if snp_occurrence == 1 & unknown
- compare differences between each method

## Parameter Optimization
- add qpcr and age as linear parameters to estimate
  - update prism2.R to write dataframe with qpcr and age as features
- fit parameters with filtered and unfiltered haplotypes

## Plots
- change color of malaria to red, asymptomatic/qPCR + to blue, asymptomatic/bloodsmear+ to green 
- ensure that "malaria" episodes (coded on plot) match up to actual diagnosed malaria episodes (consider coding this directly from the metadata, there should be a malaria y/n variable)

## Epi
-generate stats on gap length by haplotype

