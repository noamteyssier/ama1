##### Plot FOI as HouseHolds on a Map
library(tidyverse)
library(readstata13)
library(rgdal)
library(ggmap)
library(ggraph)
setwd("~/bin/ama1")

pr2 <- read_tsv("prism2/full_prism2/filtered_5pc_10r.tab") %>%
  filter(!grepl('ctrl|neg', s_Sample)) %>%
  tidyr::extract(s_Sample, into=c('date','cohortid'), regex="([:graph:]{10}\\b)-([:alnum:]{4}\\b)") %>%
  mutate(
    date = lubridate::ymd(date),
    cohortid = as.numeric(cohortid)
  )
cmeta <- read.dta13("prism2/stata/allVisits.dta") %>%
  mutate(
    date = lubridate::ymd(date),
    cohortid = as.numeric(cohortid)
  )

hhLevel <- read_tsv("prism2/stata/PRISM_GPS.csv")
cid_foi <- read_tsv("prism2/data/cid_individual_foi.tab")

# create join meta with exp data and zero missing cids' foi
maplo <- pr2 %>%
  right_join(cmeta %>% select(cohortid, date, hhid)) %>%
  left_join(hhLevel) %>%
  left_join(cid_foi) %>%
  mutate(foi = ifelse(foi == 0, NA, foi)) %>%
  select(cohortid, hhid, lat, lng, foi) %>%
  unique()

# sum household foi to make only one NA per foi-free household
maplo <- maplo %>%
  group_by(hhid) %>%
  summarise(foi_sum = sum(foi, na.rm = TRUE)) %>%
  left_join(maplo) %>%
  filter(foi_sum == 0) %>%
  group_by(hhid) %>%
  mutate(h_index = row_number() - 1) %>%
  right_join(maplo) %>%
  mutate(h_index = ifelse(is.na(h_index), 0, h_index)) %>%
  filter(h_index == 0)


# load ugandan map
ugMap <- readOGR('prism2/shapefile', 'gadm36_UGA_3')
ugMap@data %>% unique()

# convert shapefile to dataframe
ugMapDf <- fortify(ugMap[c(930, 932),])

# function to flower overlapping points equally with radius
flower <- function(df, radius){
  if (dim(df)[1] == 1){
    df <- df %>%
      mutate(
        radius = radius,
        theta = pi  * 360 / 180,
        lat2 = lat,
        lng2 = lng
      )
    return(df)
  }
  t <- pi * 360 / dim(df)[1] / 180 # calculate theta multiple

  df <- df %>%
    mutate(
      radius = radius,
      theta = t * row_number(hhid),
      lat2 = lat + (radius * cos(theta)),
      lng2 = lng - (radius * sin(theta))
    )
  return(df)
}

# apply flower function across household groups
maplo <- maplo %>%
  group_by(hhid) %>%
  do(data.frame(flower(., 0.002)))


# flowers
g <- ggplot(data = ugMapDf, aes(x = long, y = lat)) +
  geom_polygon(colour = 'black', size = 0.5, fill = "white", aes(group = group)) +
  geom_point(data = maplo %>% filter(is.na(foi_sum)), size = 9, shape = 21, color = 'grey90',
    aes(x = lng, y = lat, group = NULL)) +
  geom_point(data = maplo,
    shape=21, size=2, alpha = 0.5,
    aes(x = lng2, y = lat2, group = NULL, fill=foi)) +
  geom_point(data = maplo, size = 0.5,
    aes(x = lng, y = lat, group = NULL)) +
  scale_fill_gradient2(mid = 'goldenrod2', high='firebrick4', na.value='black', name='FOI') +
  theme_graph() +
  theme(legend.position='bottom')
ggsave("prism2/plots/maploFoi.png", g, width = 12, height = 12)



# household mean foi
maplo2 <- maplo %>%
  mutate(foi = ifelse(is.na(foi), 0, foi))

maplo2 <- maplo2 %>%
  group_by(hhid) %>%
  summarise(hfoi = mean(foi)) %>%
  left_join(maplo2)

g <- ggplot(data = ugMapDf, aes(x = long, y = lat)) +
  geom_polygon(colour = 'black', size = 0.5, fill = "white", aes(group = group)) +
  geom_point(data = maplo2,
    shape=21, size=5,
    aes(x = lng, y = lat, group = NULL, fill=hfoi)) +
  theme_graph() +
  theme(legend.position='bottom') +
  scale_fill_gradientn(colors=c('pink', 'darkred'), limits = c(0.00001, 4))

ggsave("prism2/plots/maploHFoi.pdf", g, width = 8, height = 8)
