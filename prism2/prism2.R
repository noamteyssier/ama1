library(tidyverse)
library(ggpubr)
library(readstata13)
library(vegan)
library(qgraph)
library(tidygraph)
library(ggraph)

setwd("~/bin/ama1/prism2")


cohort_meta <- read.dta13("stata/allVisits.dta")
seekdeep <- read_tsv("data/filtered_prism2.tab.txt")
vcf <- read_tsv("data/filtered_pfama1.vcf")
snpdist <- read_tsv("data/filtered_snpDist.tab")
snpdb <- read_tsv("data/ama1_snpInfo.db.tab")

# extract metadata for prism2 samples
prism2 <- seekdeep %>%
  filter(!grepl('ctrl', s_Sample)) %>%
  tidyr::extract(
    s_Sample,
    into=c('date', 'cohortid'),
    regex="([[:alnum:]-]{10}\\b)-([[:alnum:]]{4}\\b)"
  ) %>%
  mutate(
    date = lubridate::ymd(date)
  )

# extract metadata for control samples
controls <- seekdeep %>%
  filter(grepl('ctrl', s_Sample)) %>%
  tidyr::extract(
    s_Sample,
    into=c('strain_combination', 'concentration'),
    regex="ctrl-38x-([[:alnum:]a-zA-Z]{3}\\b)-([[:alnum:]a-zA-Z]{2,3}\\b)"
  ) %>%
  mutate(concentration = as.numeric(gsub('[kK]', '000', concentration)))

#################################
# comparison of reads by sample #
#################################

run_summary <- seekdeep %>%
  group_by(s_Sample) %>%
  summarise(totalReads = sum(c_ReadCnt))

run_summary_plot <- ggplot(run_summary,
  aes(x = reorder(s_Sample, totalReads), y = log10(totalReads))) +
  geom_bar(stat = 'identity') +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 3))

ggsave("plots/run_summary.png", run_summary_plot, width = 10, height = 8)

########################################
# comparison of haplotypes in controls #
########################################

control_bars <- ggplot(controls,
    aes(x = strain_combination, y = c_AveragedFrac, fill = as.factor(c_clusterID))) +
  geom_bar(stat = 'identity') +
  facet_grid(concentration~s_COI, scales = 'free') +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

ggsave("plots/control_bars.png", control_bars)


################################
# haplotype~sample frequencies #
################################




##########################
# Haplotype VCF Analysis #
##########################

longform_vcf <- vcf %>%
  select(-ID, -REF, -ALT, -QUAL, -FILTER, -INFO, -FORMAT) %>%
  gather('sample', 'snp', -CHROM, -POS) %>%
  separate(
    sample,
    into=c('hid', 'population_frequency'),
    sep = "_f"
  ) %>%
  mutate(
    hid = gsub('pfama1.', '', hid),
    population_frequency = as.numeric(population_frequency)
  )

# calculate snp frequency in population and occurrence of snp across haplotypes
longform_vcf <- longform_vcf %>%
  filter(snp > 0) %>%
  group_by(POS) %>%
  summarise(
    snp_frequency = mean(population_frequency),
    snp_occurrence = n()
  ) %>%
  left_join(longform_vcf)

# snp position and frequency plot
snp_frequency <- ggplot(longform_vcf %>% filter(snp > 0), aes(x = POS, y = hid, fill = snp_frequency)) +
  geom_point(shape = 22, size = 2) +
  theme_classic() +
  theme(axis.text.y = element_text(size = 5, angle = 20)) +
  scale_fill_gradientn(
    colours = c('navyblue', 'peru', 'firebrick4'),
    breaks = c(0.01,0.1,0.3)
  ) +
  scale_x_continuous(breaks = c(seq(0,200,10)))
ggsave("plots/snp_frequency.png", snp_frequency)


# process known snp database from PlasmoDB minor allele frequencies
maf <- snpdb %>%
  mutate(
    reference_position = gsub('Pf3D7_11_v3:_', '', Location),
    reference_position = as.numeric(gsub(',','', reference_position))
  ) %>%
  select(reference_position, Minor_Allele_Frequency)

# join prism2 data with MAF db and apply bool if found in db
known_snps <- longform_vcf %>%
  mutate(
    reference_position = POS + 1294307
  ) %>%
  left_join(maf) %>%
  select(POS, Minor_Allele_Frequency, snp_occurrence) %>%
  unique() %>%
  mutate(
    known_snp = ifelse(is.na(Minor_Allele_Frequency), FALSE, TRUE)
  )

# plot densities of snp occurrence split by found/notfound
snp_occurrence_plot <- ggplot(known_snps, aes(log2(snp_occurrence), fill = known_snp)) +
  geom_density(position = 'identity', alpha = 0.8) +
  theme_classic()
ggsave("plots/snp_occurrence.png", snp_occurrence_plot, width = 10, height = 10)

###################################
# SNP Distance between Haplotypes #
###################################

hap_mat <- vcf[-seq(1,9,1)] %>%
  as.matrix()
hap_mat[hap_mat > 0] <- 1
snp_frequencies <- cbind(vcf, snpFreq=rowMeans(hap_mat)) %>%
  as.data.frame() %>%
  gather('haplotype', 'bool', -snpFreq, -POS) %>%
  filter(
    bool > 0,
    grepl('pfama1', haplotype)
  )
(snp_frequencies %>% select(POS, snpFreq) %>% unique())$snpFreq %>% hist()

snp_frequencies %>% group_by(snpFreq) %>% summarise(c = mean(snpFreq))

# gather haplotype relevant statistics
haploStats <- prism2 %>%
  group_by(h_popUID) %>%
  summarise(
    n_timesFound = n(),
    readCount = sum(c_ReadCnt),
    meanPC = mean(c_AveragedFrac),
    meanRC = mean(c_ReadCnt),
    meanCI = mean(c_clusterID)
  ) %>%
  mutate(h_popUID = as.numeric(gsub('pfama1.', '', h_popUID)))

# convert snpdist matrix to lower triangular
snpdist[upper.tri(snpdist)] <- NA

# convert triangular distance matrix to longform table
haploGraph <- snpdist %>%
  gather('haplotype_2', 'steps', -haplotype) %>%
  filter(!is.na(steps)) %>%
  mutate(
    haplotype = gsub('pfama1.','',haplotype),
    haplotype = as.numeric(gsub('_f[[:alnum:].]+','', haplotype)),
    haplotype_2 = gsub('pfama1.','',haplotype_2),
    haplotype_2 = as.numeric(gsub('_f[[:alnum:].]+','', haplotype_2))
  )

# convert to table_graph object
hg <- as_tbl_graph(haploGraph) %>%

  # node based dplyr
  activate(nodes) %>%
  left_join(haploStats %>% mutate(h_popUID = as.character(h_popUID)), by = c('name' = 'h_popUID')) %>%
  mutate(shapeBool = ifelse(meanRC < 1000, 'tri', 'circle')) %>%

  # edge based dplyr
  activate(edges) %>%
  filter(steps < 3) %>%
  mutate(fixedAlpha = ifelse(steps == 1, 1, 0.75)) # fixed alpha size (gray 1 steps)


ggraph(hg) +
  geom_edge_fan(aes(width = as.factor(steps),  alpha = as.factor(fixedAlpha))) +
  theme_graph() +
  scale_edge_width_discrete(range = c(1,0.75)) +
  scale_edge_alpha_discrete(range = c(0.5, 1)) +
  # geom_node_label(aes(label = numPatients), nudge_x = 0.25, nudge_y = 0.25, size = 2.5) +
  geom_node_point(aes(size = meanPC, fill = minFreq, shape = shapeBool))




#########################
# Visit Parasite Status #
#########################
cohort_meta %>% colnames() %>% as.tibble()

# convert long strings to bools
cohort_meta <- cohort_meta %>%
  select(cohortid, date, fever, qpcr, qPCRdich, parasitedensity) %>%
  mutate(cohortid = as.character(cohortid)) %>%
  mutate(malariaCall = case_when(
      parasitedensity > 0  ~ TRUE, # if pdensity is positive
      TRUE ~ FALSE # default case
    )
  ) %>%
  mutate(fever = case_when(
      fever == 'no' ~ FALSE,
      fever == 'yes' ~ TRUE
    )
  )

# create fill status and colour status
cohort_meta <- cohort_meta %>%
  mutate(fill_status = case_when(
    fever == TRUE & malariaCall == TRUE ~ 'parasite_positive',
    fever == FALSE & malariaCall == TRUE ~ 'parasite_positive',
    fever == FALSE & qPCRdich == 1 ~ 'parasite_positive',
    TRUE ~ 'parasite_negative'
    )) %>%
  mutate(colour_status = case_when(
    fever == TRUE & malariaCall == TRUE ~ 'fever-bloodsmear+',
    fever == FALSE & malariaCall == TRUE & qPCRdich == 1 ~ 'asymp-bloodsmear+',
    fever == FALSE & malariaCall == FALSE & qPCRdich == 1 ~ 'asymp-qpcr+',
    TRUE ~ 'qpcr-'
    ))

# merge prism2 data with cohort meta
prism2 <- prism2 %>%
  inner_join(cohort_meta)

# calculate number of infection events by cohortid
prism2 <- prism2 %>%
  select(date, cohortid) %>%
  unique() %>%
  group_by(cohortid) %>%
  summarise(
    infection_events = n()
  ) %>%
  left_join(prism2)

prism2 <- prism2 %>%
  mutate(
    hap_qpcr = c_AveragedFrac * qpcr,
    shapeBool = ifelse(fill_status == 'parasite_negative', 1, 0)
  )

####################################
# CID Over time  // Timeline Plots #
####################################

# plot timelines without haplotype data
patient_timeline <- ggplot(data = cohort_meta %>%
  filter(!is.na(qPCRdich)),
  aes(x = date, y = cohortid)) +
  geom_point(shape = 21, size = 2, aes(fill = as.factor(colour_status), alpha = as.factor(fill_status))) +
  theme_classic()
ggsave('plots/patientTimelines.png', patient_timeline, width = 10, height = 12)

# haplodrop
haplodrop <- ggplot(data = prism2 %>% filter(infection_events > 2)) +

  # haplotype squares
  geom_point(shape = 22,
    aes(x = date, y = factor(h_popUID), size = log10(hap_qpcr), fill = factor(h_popUID))) +

  # timeline points
  geom_point(size = 3, alpha = 0.4, aes(
    x = date,
    y = 'visit',
    shape = as.factor(shapeBool),
    colour = as.factor(interaction(fill_status, colour_status)))) +

  # theme and facets
  facet_wrap(~cohortid, scale = 'free_y') +
  theme_classic() +
  guides(fill=FALSE,
    colour = guide_legend(title = 'Visit Type'),
    size = guide_legend(title = 'qPCR (log10)'),
    shape=F
    ) +
  scale_colour_manual(values = c('forestgreen','royalblue','red','peru','black')) +
  scale_shape_manual(values = c(16, 21)) +
  scale_y_discrete(expand= c(0,1)) +
  theme(aspect.ratio = 1.5,
    axis.text.y =  element_text(angle = 30)) +
  labs(x = 'Date', y = 'Haplotype Population', title = 'Haplotype Timelines')
ggsave("plots/haplodrop.png", haplodrop, width = 25, height = 12)
