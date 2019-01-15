library(tidyverse)
library(ggpubr)
setwd("~/bin/ama1/prism2")


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
