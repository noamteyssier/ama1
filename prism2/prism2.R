library(tidyverse)
library(ggpubr)
library(readstata13)

setwd("~/bin/ama1/prism2")


cohort_meta <- read.dta13("stata/allVisits.dta") %>%
  select(cohortid, hhid, date, qpcr) %>%
  mutate(cohortid = as.character(cohortid))
seekdeep <- read_tsv("data/pfama1_sampInfo.tab.txt")


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

controls <- seekdeep %>%
  filter(grepl('ctrl', s_Sample))
control_bars <- ggplot(controls,
    aes(x = s_Sample, y = c_AveragedFrac, fill = as.factor(c_clusterID))) +
  geom_bar(stat = 'identity') +
  facet_grid(~s_COI, scales = 'free') +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

ggsave("plots/control_bars.png", control_bars)


################################
# haplotype~sample frequencies #
################################

ggplot(seekdeep %>% select(h_popUID, h_SampCnt) %>% unique(),
    aes(x = h_popUID, y = h_SampCnt)) +
  geom_bar(stat = 'identity')

ggplot(seekdeep, aes(x = h_popUID, y = h_SampFrac)) +
  geom_point()

ggplot(seekdeep, aes(h_SampFrac)) +
  geom_density(fill = 'peru')

seekdeep %>%
  select(h_popUID) %>%
  unique() %>%
  dim()

seekdeep %>%
  filter(h_SampFrac > 0.01) %>%
  select(h_popUID) %>%
  unique() %>%
  dim()


####################################
# CID Over time  // Timeline Plots #
####################################

sdf <- seekdeep %>%
  filter(grepl('20', s_Sample)) %>%
  tidyr::extract(
    s_Sample,
    into = c("date", "cid"),
    regex = "([[:alnum:]-]{10}\\b)-([[:alnum:]]{4}\\b)"
  ) %>%
  mutate(date = lubridate::ymd(date))

sdf <- sdf %>%
  group_by(cid) %>%
  summarise(ph_mean = mean(s_COI)) %>%
  left_join(sdf)

sdf <- sdf %>%
  select(cid, date) %>%
  unique() %>%
  group_by(cid) %>%
  summarise(p_visits = n()) %>%
  left_join(sdf)

sdf <- sdf %>%
  left_join(cohort_meta, by=c('date', 'cid' = 'cohortid')) %>%
  mutate(
    hap_density = c_AveragedFrac * qpcr,
    h_popUID = gsub('pfama1.', '', h_popUID)
  )

ggplot(sdf %>% filter(ph_mean > 2) %>% filter(p_visits > 1), aes(x = date, y = h_popUID)) +
  geom_point(aes(size = hap_density), shape = 22) +
  facet_wrap(~cid, scales = 'free')
