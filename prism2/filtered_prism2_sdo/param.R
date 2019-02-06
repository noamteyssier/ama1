library(tidyverse)

setwd("~/bin/ama1/prism2/filtered_prism2_sdo")

param <- read_tsv("param_var.tab.txt")
pattern <- read_tsv("pattern_var.tab.txt")

pattern %>%
  select(-count) %>%
  spread(key = filter, value = frequency)

ggplot(pattern, aes(x = pattern, y = frequency)) +
  geom_boxplot() +
  geom_jitter(shape = 21, aes(fill = filter))+
  theme_classic()

param
