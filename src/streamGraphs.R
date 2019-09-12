library(tidyverse)
library(streamgraph)
library(RColorBrewer)

sg_add_marker <- function(sg, x, label="", stroke_width=0.5, stroke="#7f7f7f", space=5,
                          y=0, color="#7f7f7f", size=12, anchor="start") {
  
  if (inherits(x, "Date")) { x <- format(x, "%Y-%m-%d") }
  
  mark <- data.frame(x=x, y=y, label=label, color=color, stroke_width=stroke_width, stroke=stroke,
                     space=space, size=size, anchor=anchor, stringsAsFactors=FALSE)
  
  if (is.null(sg$x$markers)) {
    sg$x$markers <- mark
  } else {
    sg$x$markers <- bind_rows(mark, sg$x$markers)
  }
  
  sg
  
}

select_cid <- function(cid){
  # Function to select a cid from the given data
  meta %>% 
    filter(cohortid == cid) %>%
    left_join(sdo) %>%
    select(cohortid, date, visittype, qpcr, malariacat, h_popUID, c_AveragedFrac) %>%
    mutate(
      qpcr_log = log10(qpcr), 
      qpcr_hap_log = qpcr_log*c_AveragedFrac
    ) %>% 
    return()
}
plot_stream <- function(table, save=FALSE, offset='silhouette', interpolate='cardinal'){
  # Function to plot a stream graph
  
  sg <- streamgraph(
    table, date=date, value=qpcr_hap_log, 
    key=h_popUID, offset=offset, interpolate = interpolate
  ) %>% 
    sg_fill_brewer("Set2") %>% 
    add_dates(table)
  
  return(sg)
}
add_dates <- function(sg, table, y = -0.06){
  
  # isolate date~conditions and add colorscheme
  table_dates <- table %>% 
    group_by(date, malariacat, qpcr) %>%
    summarise(minimum_date_qpcr = max(qpcr)) %>% 
    unique() %>% 
    left_join(color_conditions) %>% 
    mutate(
      fill = ifelse(minimum_date_qpcr == 0, circle_types[1], circle_types[2])
    )
  
  # iterate through dates and plot points colored by condition
  for (i in seq(1, dim(table_dates)[1])){
    
    # isolate date~condition as row for dataframe
    visit <- table_dates[i,]
    
    # add to streamgraph
    sg <- sg %>% 
      sg_add_marker(
        x = visit$date, y = y,
        anchor='middle',
        label=visit$fill,
        color=visit$color
      )
    
  }
  
  return(sg)
}

# Load meta and seekdeep output 
setwd("~/projects/ama1/src")

meta <- read_tsv("../prism2/stata/full_meta_6mo_fu.tab") %>%
  filter(!is.na(qpcr))
sdo <- read_tsv("../prism2/full_prism2/final_filter.tab")

# give color conditions and fill status
color_conditions <- meta %>% select(malariacat) %>% unique()
color_conditions$color <- c('blue', 'green', 'red')
circle_types <- c("\u25CB" ,"\u25CF")


# select cid with function, pipe into streamgraph
stream <- select_cid(3614) %>% plot_stream(interpolate='basis-open')
stream

# 3604 doesn't work...
cids <- c(3079, 3164, 3614)
