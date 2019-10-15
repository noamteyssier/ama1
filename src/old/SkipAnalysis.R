library(tidyverse)
library(ggridges)

skip_frame <- read_tsv("../skips.tab")
sdo <- read_tsv("../prism2/full_prism2/final_filter.tab")

skip_frame <- skip_frame %>%
  group_by(cohortid, h_popUID) %>%
  summarise(first_visit = min(visit_num)) %>%
  left_join(skip_frame) %>%
  filter(first_visit != visit_num)


haplotype_plaf <- sdo %>%
  select(cohortid, h_popUID) %>%
  unique() %>%
  group_by(h_popUID) %>%
  count()
haplotype_plaf <- haplotype_plaf %>%
  mutate(plaf = n / haplotype_plaf$n %>% sum())

skips_plaf <- skip_frame %>%
  left_join(haplotype_plaf)


# histogram of skips

ggplot(skips_plaf %>% group_by(skips) %>% summarise(num = n()), aes(x = skips, y = num)) +
  geom_bar(stat='identity') +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0,21,1))

# histogram population allele frequencies by skips
ggplot(skips_plaf, aes(x = plaf, y = as.factor(skips), fill=as.factor(skips))) +
  geom_density_ridges() +
  theme_minimal() +
  guides(fill = F)

# scatter plot of number of skips against population allele_frequency
ggplot(skips_plaf, aes(x = skips, y = plaf)) +
  geom_jitter()
