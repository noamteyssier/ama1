library(tidyverse)


plot_durations <- function(frame){
  ggplot(frame, aes(x = grouping, y = estimate, color=ie_type)) +
    geom_errorbar(
      aes(ymin = ci_05, ymax = ci_95), 
      stat='identity', width=0.4,
      position=position_dodge2(width=0.5)
    ) +
    geom_point(
      size=2.5, aes(pch=ie_type),
      position=position_dodge2(width=0.4)
    ) +
    coord_flip() +
    scale_color_brewer(palette='Dark2') +
    theme_classic()
}

apply_filter <- function(frame, categories){
  filt <- frame %>% 
    filter(grouping %in% categories)
  
  filt$grouping <- factor(filt$grouping, levels = rev(categories))
  
  filt
}


##########
## main ##
##########

setwd("~/projects/ama1/src/")

durations <- read_tsv("durations.tab")


categories_plot_a <- c(
  'overall', 'Female', 'Male', 
  '< 5 years', '5-15 years', '16 years or older'
) 

categories_plot_b <- c(
  "('Female', '< 5 years')", "('Male', '< 5 years')",
  "('Female', '5-15 years')", "('Male', '5-15 years')", 
  "('Female', '16 years or older')", "('Male', '16 years or older')",
  "('baseline', 'Female')", "('baseline', 'Male')", 
  "('new infection', 'Female')", "('new infection', 'Male')"
)

durations_a <- apply_filter(durations, categories_plot_a)
durations_b <- apply_filter(durations, categories_plot_b)

plot_durations(durations_a)
ggsave("../plots/durations/errorbar_durationsOverall.pdf")

plot_durations(durations_b)            
ggsave("../plots/durations/errorbars_durationsGender.pdf")
