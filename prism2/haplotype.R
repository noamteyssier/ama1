library(tidyverse)
library(readstata13)

setwd("~/bin/ama1/prism2")

sample_meta <- read.dta13("stata/allVisits.dta") %>%
  mutate(cohortid = as.character(cohortid)) %>%
  select(cohortid, hhid, date, qpcr)
pr2 <- read_tsv("data/pfama1_sampInfo.tab.txt") %>%
  extract(
    s_Sample,
    into=c('date', 'cid'),
    regex="([[:alnum:]-]{10}\\b)-([[:alnum:]]{4}\\b)"
  ) %>%
  mutate(date = lubridate::ymd(date)) %>%
  left_join(sample_meta, by = c('date', 'cid' = 'cohortid'))

pr2 %>%
  filter(h_popUID == 'pfama1.00')

pr2[pr2$h_popUID == 'pfama1.00',]$c_AveragedFrac %>% hist()


########################################################
# human haplotype population differentiation with qpcr #
########################################################

human <- c(as.character('00'),as.character(46),as.character(51),as.character(50),as.character(58),as.character(59),as.character(61),as.character(62),as.character(63),as.character(66),as.character(69),as.character(72),as.character(74),as.character(81),as.character(82),as.character(90),as.character(91))
hapcat_df <- pr2 %>%
  mutate(
    popnum = gsub('pfama1.','',h_popUID),
    hapcat = ifelse(popnum %in% human, 'human','pf')
  )

# qpcr densities
ggplot(hapcat_df, aes(log10(qpcr), fill = hapcat)) +
  geom_density(position = 'identity', alpha = 0.5)

# readcount densities
ggplot(hapcat_df, aes(log10(c_ReadCnt), fill = hapcat)) +
  geom_density()

# human reads take up entirety of sample
ggplot(hapcat_df, aes(c_AveragedFrac, fill = hapcat)) +
  geom_density(position = 'identity', alpha = 0.5)
