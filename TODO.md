# TODO for AMA1

## Summarise Run
- number of chimeras per sample
- list of everything with zero reads

## snp_frequency
- sum instead of mean?

## Haplotype Filtering
- ~~filter haplotypes with OOSSLF method~~
  - ~~one off same sample low frequency~~
- ~~filter if snp_occurrence == 1 & unknown~~
- separate LFH into population and sample frequencies
- add ooslfh
- for one off samples don't penalize the higher percentage one off haplotype
    - in case where both are under threshold, keep top one
- create flags to validate haplotypes and check to see if filtering excludes them
- for haplotype in population use occurence instead of read fraction
- separate flags for haplotype filtering

## Parameter Optimization
- add qpcr and age as linear parameters to estimate
- fit parameters with filtered and unfiltered haplotypes

## Plots
- change color of malaria to red, asymptomatic/qPCR + to blue, asymptomatic/bloodsmear+ to green
- ensure that "malaria" episodes (coded on plot) match up to actual diagnosed malaria episodes (consider coding this directly from the metadata, there should be a malaria y/n variable)

## Epi
- generate stats on gap length by haplotype

## Meeting Notes
- check if replicates function properly when one of the pairs has zero reads with SeekDeep
- check to see if the expected haplotype is found in the 7 strain controls that only show 5
