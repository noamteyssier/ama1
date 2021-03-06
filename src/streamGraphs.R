library(tidyverse)
library(streamgraph)
library(htmlwidgets)
library(webshot)
library(viridis)
library(progress)

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
    left_join(sdo, by = c('cohortid', 'date')) %>%
    select(cohortid, date, visittype, qpcr, malariacat, h_popUID, c_AveragedFrac) %>%
    mutate(
      qpcr_log = log10(qpcr), 
      qpcr_hap_log = qpcr_log*c_AveragedFrac
    ) %>% 
    return()
}
plot_stream <- function(table, offset='silhouette', interpolate='cardinal', width=NULL, height=NULL){
  # Function to plot a stream graph
  num_haps <- table$h_popUID %>% unique() %>% length()
  sg <- streamgraph(
    table, date=date, value=qpcr_hap_log, 
    key=h_popUID, offset=offset, interpolate = interpolate,
    width = width, height = height
  ) %>% 
    sg_fill_manual(
      values = viridisLite::viridis(num_haps, end=0.8)
    ) %>% 
    add_dates(table)
  
  return(sg)
}
add_dates <- function(sg, table){
  
  # isolate date~conditions and add colorscheme
  table_dates <- table %>% 
    group_by(date, malariacat, qpcr) %>%
    summarise(minimum_date_qpcr = max(qpcr)) %>% 
    unique() %>% 
    left_join(color_conditions, by='malariacat') %>% 
    mutate(
      fill = ifelse(minimum_date_qpcr == 0, circle_types[1], circle_types[2]),
      stroke_width = ifelse(malariacat == 'Malaria', 2, 0.5),
      stroke_color = ifelse(malariacat == 'Malaria', 'red', 'black')
    )
  
  # set timeline at highest point + offset determined from data
  y = (sg$x$data %>% group_by(date) %>% summarise(s = sum(value)))$s %>% max()
  offset = (y / 5) * 0.01
  y = y + offset
  
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
        color=visit$color,
        stroke_width = visit$stroke_width,
        stroke = visit$stroke_color
      )
    
  }
  
  return(sg)
}
stream_over_set <- function(cid_set, prefix=NULL){
  # function to plot a streamgraph + timeline over a set of cids
  
  # add prefix if given
  if (!is.null(prefix)){
    prefix = paste0(prefix, '_')
  }
  
  pb <- progress_bar$new(
    format = "  generating streamgraphs for set [:bar] :percent eta: :eta",
    total = length(cid_set), clear = FALSE, width= 60)

  
  # plot and save for each cid
  for (c in cid_set){
    pb$tick()
    html_name = paste0(prefix, c, '.html')
    png_name = paste0(prefix, c, '.png')
    
    s <- select_cid(c) %>% plot_stream(interpolate='basis')
    saveWidget(s, html_name)
    webshot(html_name, png_name)
  }
}


# Load meta and seekdeep output 
setwd("~/projects/ama1/src")

meta <- read_tsv("../prism2/stata/full_meta_6mo_fu.tab") %>%
  filter(!is.na(qpcr))
sdo <- read_tsv("../prism2/full_prism2/final_filter.tab") %>% 
  filter(cohortid %in% meta$cohortid)

meta <- meta %>% filter(cohortid %in% sdo$cohortid)

# give color conditions and fill status
color_conditions <- meta %>% select(malariacat) %>% unique()
color_conditions$color <- c('blue', 'green', 'red')
circle_types <- c("\u25CB" ,"\u25CF")

# select cid with function, pipe into streamgraph
stream <- select_cid(3604) %>% plot_stream(interpolate='basis')


cid_complexity <- sdo %>% 
  group_by(cohortid, date) %>% 
  select(cohortid, date, s_COI) %>% 
  group_by(cohortid) %>% 
  summarise(m = max(s_COI))


# high complexity
high_complexity_cids <- filter(cid_complexity, m > 3)$cohortid
stream_over_set(high_complexity_cids, prefix='hCOI')

# malaria positive
malaria_positive_cids <- filter(meta, malariacat == 'Malaria')$cohortid %>%  unique()
stream_over_set(malaria_positive_cids, prefix='mPositive')

# full set
full_set_cids <- unique(sdo$cohortid)
stream_over_set(full_set_cids, prefix='cid')

