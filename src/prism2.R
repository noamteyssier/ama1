library(tidyverse)
library(readstata13)
library(tidygraph)
library(ggraph)

setwd("~/bin/ama1/prism2")


cohort_meta <- read.dta13("stata/allVisits.dta")
seekdeep <- read_tsv("full_prism2/pfama1_sampInfo.tab.txt")
vcf <- read_tsv("full_prism2/pfama1.vcf", skip = 3)
snpdist <- read_tsv("full_prism2/pfama1.dist")
snpdb <- read_tsv("data/ama1_snpInfo.db.tab")
readcounts <- read_tsv("data/readCounts.tab")

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
    into=c('cycles','strain_combination', 'concentration'),
    regex="ctrl-([[:alnum:]]{3}\\b)-([[:alnum:]]{3}\\b)-([[:alnum:]]{2,3}\\b)"
  ) %>%
  mutate(
    concentration = as.numeric(gsub('[kK]', '000', concentration)),
    cycles = as.numeric(gsub('x','', cycles))
  )

#################################
# comparison of reads by sample #
#################################

run_summary <- seekdeep %>%
  group_by(s_Sample) %>%
  summarise(totalReads = sum(c_ReadCnt)) %>%
  mutate(
    count_bin = case_when(
      totalReads == 0 ~ "0",
      totalReads <= 10 ~ ">0",
      totalReads <= 100 ~ ">10",
      totalReads <= 1000 ~ ">100",
      totalReads <= 10000 ~ ">1000",
      TRUE ~ ">10000"
    )
  )

run_summary_plot <- ggplot(run_summary,
  aes(x = reorder(s_Sample, totalReads), y = log10(totalReads))) +
  geom_bar(stat = 'identity') +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, size = 3))
ggsave("plots/run_summary.png", run_summary_plot, width = 10, height = 8)

cid_date_bar <- ggplot(run_summary %>%
    filter(grepl('20', s_Sample)) %>%
    group_by(count_bin) %>%
    summarise(c = n()),
  aes(x = as.factor(count_bin), y = c)) +
  geom_bar(stat = 'identity') +
  theme_classic() +
  xlab("Read Counts") +
  ylab("Number of cid~date")
ggsave("plots/sample_binned_readCounts.png", cid_date_bar, width = 8, height = 8)

#############################################
# comparison of read counts by sample fastq #
#############################################

rc <- readcounts %>%
  filter(grepl('20', sample)) %>%
  tidyr::extract(
    sample,
    into=c('date','cid','plate','rep'),
    regex="([[:graph:]]{10}\\b)-([[:alnum:]]{4}\\b)-PR2-([[:alnum:]]{3,4}\\b)-([[:alnum:]])"
  ) %>%
  gather('filter_status', 'count', -date, -cid, -plate, -rep )

binned <- rc %>%
  mutate(
    count_bin = case_when(
      count == 0 ~ " 0 ",
      count <= 10 ~ ">0",
      count <= 100 ~ ">10",
      count <= 1000 ~ ">100",
      count <= 10000 ~ ">1000",
      TRUE ~ ">10000"
    )
  ) %>%
  group_by(filter_status, count_bin) %>%
  summarise(num_bin = n())

binned <- binned %>%
  group_by(filter_status) %>%
  summarise(total = sum(num_bin)) %>%
  left_join(binned) %>%
  mutate(pc_bin = num_bin / total)

# bar plot of binned read count percentages
binned_percentages <- ggplot(binned, aes(x = as.factor(count_bin), y = pc_bin, fill = filter_status)) +
  geom_bar(stat = 'identity', position = 'dodge') +
  scale_fill_manual(values = c('firebrick4', "burlywood4")) +
  theme_classic() +
  ylab("Percentage of Samples") +
  xlab("Read Count")
ggsave("plots/fastq_binned_readCounts.png", binned_percentages, width = 8, height = 8)

# density plot of read counts filtered/unfiltered
read_count_densities <- ggplot(rc %>% mutate(count = ifelse(count == 0, 0.1, count)),
    aes(log10(count), fill = filter_status)) +
  geom_density(position = 'identity', alpha = 0.5) +
  scale_fill_manual(values = c('cadetblue4', "firebrick4")) +
  xlab("Density of read counts pre/post filtering (log10)") +
  ylab("Percent of Samples") +
  theme_classic()
ggsave("plots/read_densities.png", read_count_densities, width = 10, height = 8)



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
colnames(vcf)[1] <- 'CHROM' # removing commenting hashtag for colnames
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

# quick join for haplotype and positional frequencies of snps found in hap
haplotype_positional_frequency <- longform_vcf %>%
  filter(snp > 0) %>%
  select(-snp, -CHROM) %>%
  mutate(hid = as.numeric(hid))


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
num_haps <- (longform_vcf %>% select(hid) %>% unique() %>% dim())[1]
snp_occurrence_plot <- ggplot(known_snps, aes(log10(snp_occurrence / num_haps), fill = known_snp)) +
  geom_density(position = 'identity', alpha = 0.8) +
  xlab("percentage of haplotypes snp found in (log10)") +
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
  mutate(shapeBool = ifelse(meanRC < 1000, '<1000rc', '>1000rc')) %>%
  filter(!is.na(shapeBool)) %>%

  # edge based dplyr
  activate(edges) %>%
  filter(steps < 2) %>%
  mutate(fixedAlpha = ifelse(steps == 1, 1, 0.75)) # fixed alpha size (gray 1 steps)

hg %>% activate(nodes) %>% filter(is.na(shapeBool))

g <- ggraph(hg) +
  theme_graph() +
  geom_edge_fan(aes(width = as.factor(steps),  alpha = as.factor(fixedAlpha))) +
  geom_node_point(aes(size = meanPC, fill = meanCI, shape = shapeBool)) +
  geom_node_label(aes(label = n_timesFound), nudge_x = 0.25, nudge_y = 0.25, size = 2.5) +
  scale_edge_width_discrete(range = c(1,0.75)) +
  scale_edge_alpha_discrete(range = c(0.5, 1)) +
  scale_alpha(range = c(1,1)) +
  scale_shape_manual(values = c(21,24))  +
  scale_fill_gradientn(colours = c('peru', 'red', 'navy') %>% rev())

cairo_pdf(filename = "plots/haploGraph.pdf", width = 10, height = 10)
plot(g)
dev.off()

####################################
# Haplotype Position Differentials #
####################################

# function to find positions of difference between two haplotypes
snp_diff <- function(df, hap_pos_freq){
  # df = [h1, h2, steps]
  # hap_pos_freq = [POS, snp_frequency, hid]
  # output = [POS, h1, h2, snp_frequency]

  pos_snp_freq <- haplotype_positional_frequency %>%
    select(POS, snp_frequency) %>%
    unique()

  pos_diff <- hap_pos_freq %>%
    filter(hid %in% c(df[1], df[2])) %>%
    select(POS, hid) %>%
    group_by(POS) %>%
    summarise(count = n()) %>%
    filter(count == 1) %>%
    select(-count) %>%
    mutate(
      h1 = as.numeric(df[1]),
      h2 = as.numeric(df[2]),
      steps = as.numeric(df[3])
    ) %>%
    left_join(pos_snp_freq, by = 'POS')

  return (pos_diff)
}

positional_haplotype_differences <- apply(haploGraph, 1, snp_diff, haplotype_positional_frequency) %>%
  bind_rows()

step_frequency_density <- ggplot(positional_haplotype_differences %>% filter(steps < 5), aes(x = snp_frequency, fill = as.factor(steps))) +
  geom_density(alpha = 0.5)
ggsave("plots/stepFreqDensity.png", step_frequency_density, width = 10, height = 8)

to_remove <- haploGraph %>%
  left_join(
    positional_haplotype_differences,
    by = c('haplotype' = 'h1', 'haplotype_2' = 'h2', 'steps')
  ) %>%
  filter(snp_frequency < 0.05) %>%
  select(haplotype, haplotype_2) %>%
  unique()

filt_haploGraph <- haploGraph %>%
  anti_join(to_remove)

filt_hg <- as_tbl_graph(filt_haploGraph) %>%

  # node based dplyr
  activate(nodes) %>%
  left_join(haploStats %>% mutate(h_popUID = as.character(h_popUID)), by = c('name' = 'h_popUID')) %>%
  mutate(shapeBool = ifelse(meanRC < 1000, '<1000rc', '>1000rc')) %>%
  filter(!is.na(shapeBool)) %>%

  # edge based dplyr
  activate(edges) %>%
  filter(steps < 2) %>%
  mutate(fixedAlpha = ifelse(steps == 1, 1, 0.75)) # fixed alpha size (gray 1 steps)

filt_g <- ggraph(filt_hg) +
  theme_graph() +
  geom_edge_fan(aes(width = as.factor(steps),  alpha = as.factor(fixedAlpha))) +
  geom_node_point(aes(size = meanPC, fill = meanCI, shape = shapeBool)) +
  geom_node_label(aes(label = n_timesFound), nudge_x = 0.25, nudge_y = 0.25, size = 2.5) +
  scale_edge_width_discrete(range = c(1,0.75)) +
  scale_edge_alpha_discrete(range = c(0.5, 1)) +
  scale_alpha(range = c(1,1)) +
  scale_shape_manual(values = c(21,24))  +
  scale_fill_gradientn(colours = c('peru', 'red', 'navy') %>% rev())

cairo_pdf(filename = "plots/filt_haploGraph.pdf", width = 10, height = 10)
plot(filt_g)
dev.off()


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
