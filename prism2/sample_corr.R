library(tidyverse)
library(readstata13)
setwd("~/bin/ama1/prism2")

sample_meta <- read.dta13('stata/allVisits.dta') %>%
  mutate(cid = as.character(cohortid)) %>%
  select(cid, date, qpcr)

pr2 <- read_tsv("data/flagstat.tab") %>%
  filter(grepl(20, sample)) %>%
  extract(
    sample,
    into=c('date', 'cid', 'plate', 'rep'),
    regex="([[:alnum:]-]{10}\\b)-([[:alnum:]]{4}\\b)-PR2-([[:alnum:]]{3,4}\\b)-([ABab])"
  ) %>%
  mutate(date = lubridate::ymd(date)) #%>%
  #left_join(sample_meta, by=c('date', 'cid'))

pr2 %>%
  filter(cid == '3093')

sample_meta %>%
  filter(cid == '3510')

pr2 %>%
  select(date, cid) %>%
  unique() %>%
  anti_join(sample_meta)

sample_meta %>%
  select(date, cid, qpcr) %>%
  filter(qpcr > 0) %>%
  left_join(pr2) %>%
  filter(is.na(total)) %>%
  filter(cid %in% pr2$cid)
