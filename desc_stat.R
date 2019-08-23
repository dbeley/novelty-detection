rm(list=ls())
setwd("/home/david/Documents/Données")

library("dplyr")
library("ggplot2")
library("forcats")

theme_set(theme_minimal() + theme(text = element_text(size = 12)))

nytdata <- read.csv("experimentations.csv", sep="\t")
datapapers <- read.csv("datapapers.csv", sep="\t")

summary(nytdata)
names(nytdata)
nrow(nytdata)
  
# TODO
nytdata %>%
  group_by(year, principal_classifier) %>%
  summarize(n = n()) %>%
  ggplot(aes(x=year, y=n, group=principal_classifier)) +
  geom_line(aes(color=principal_classifier)) +
  labs(title = "", color="Catégorie", y="Catégorie", x="Année")

nytdata %>%
  ggplot(aes(x=principal_classifier)) +
  geom_bar(aes(fill=principal_classifier)) +
  labs(title ="", fill="Catégorie", x="Nbr documents", "Catégorie")

head(nytdata$texts)

names(datapapers)
nrow(datapapers)
datapapers %>%
  group_by(year, theme) %>%
  summarize(n = n()) %>%
  ggplot(aes(x=year, y=n, group=theme)) +
  geom_line(aes(color=theme)) +
  labs(title = "", color="Catégorie", y="Catégorie", x="Année")

datapapers %>%
  ggplot(aes(x=theme)) +
  geom_bar(aes(fill=theme)) +
  labs(title ="", fill="Catégorie", x="Nbr documents", "Catégorie")

nytdata$texts <- gsub("[^[:alpha:]||[:blank:]]", " ", nytdata$texts)

nvec <- length(df)
breaks <- which(! nzchar(df))
nbreaks <- length(breaks)
if (breaks[nbreaks] < nvec) {
  breaks <- c(breaks, nvec + 1L)
  nbreaks <- nbreaks + 1L
}
if (nbreaks > 0L) {
  df <- mapply(function(a,b) paste(df[a:b], collapse = " "),
               c(1L, 1L + breaks[-nbreaks]),
               breaks - 1L)
}


my.data <- data.frame(nytdata$texts[1])
paste(my.data[,1], collapse = "|")
