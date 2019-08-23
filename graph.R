library('tidyverse')

theme_set(theme_minimal() + theme(text=element_text(size=18)))

datapapers <- read.csv("~/Documents/Données/datapapers.csv", sep='\t')

head(datapapers)
View(datapapers)

datapapers %>%
  group_by(theme) %>%
  summarize(count = n()) %>%
  ggplot(aes(x=theme, y=count, fill=theme)) +
  geom_bar(stat='identity') +
  ylab("Nombre de papiers") +
  xlab("Catégorie") +
  theme(legend.position="none")

experimentations <- read.csv("~/Documents/Données/experimentations.csv", sep='\t')

experimentations %>%
  group_by(principal_classifier) %>%
  summarize(count=n()) %>%
  ggplot(aes(x=principal_classifier, y=count, fill=principal_classifier)) +
  geom_bar(stat='identity') +
  ylab("Nombre d'articles") +
  xlab("Catégorie") +
  theme(legend.position='none')


résultats_mediamining <- read.csv("/home/david/Documents/Stage/sentences_embeddings/Exports/Résultats_condensés_mediamining.csv", sep=';')
résultats <- read.csv("/home/david/Documents/Stage/sentences_embeddings/Exports/Résultats_condensés.csv", sep=';')

names(résultats)

# USE 2018
# fasttext 2015
# sent2vec 2018
# infersent 2017
résultats <- résultats %>%
  mutate(encoder_name = fct_recode(encoder_name,
                                   "sent2vec (2018)" = "sent2vec",
                                   "USE (2018)" = "USE",
                                   "fasttext (2015)" = "fasttext",
                                   "infersent (2017)" = "infersent"))
 
résultats_mediamining <- résultats_mediamining %>%
  mutate(encoder_name = fct_recode(encoder_name,
                                   "sent2vec (2018)" = "sent2vec",
                                   "USE (2018)" = "USE",
                                   "fasttext (2015)" = "fasttext",
                                   "infersent (2017)" = "infersent"))

résultats %>%
  # mutate(encoder_name = as.factor(encoder_name)) %>%
  group_by(encoder_name) %>%
  filter(data_filename == "datapapers_clean.csv") %>%
  filter(encoder_name != "tf-idf") %>%
  # filter(data_filename == "experimentations.csv") %>%
  summarize(time = mean(as.numeric(temps))) %>%
  # View
  ggplot(aes(x=fct_reorder(encoder_name, time, .desc=FALSE), y=time, fill=encoder_name)) +
  geom_bar(stat='identity') +
  ylab("Moyenne itération (s)") +
  xlab("Encodeur") +
  theme(legend.position='none')

résultats_mediamining %>%
  group_by(encoder_name) %>%
  # filter(data_filename == "datapapers_clean.csv") %>%
  filter(data_filename == "experimentations.csv") %>%
  summarize(time = mean(as.numeric(temps))) %>%
  # View
  ggplot(aes(x=fct_reorder(encoder_name, time, .desc=FALSE), y=time, fill=encoder_name)) +
  geom_bar(stat='identity') +
  ylab("Moyenne itération (s)") +
  xlab("Encodeur") +
  theme(legend.position='none')

résultats_bruts <- read.csv("/home/david/Documents/Stage/sentences_embeddings/Exports/Résultats_bruts.csv", sep=';')
names(résultats_bruts)
résultats_bruts <- résultats_bruts %>%
  mutate(encoder_name = fct_recode(encoder_name,
                                   "sent2vec (2018)" = "sent2vec",
                                   "USE (2018)" = "USE",
                                   "fasttext (2015)" = "fasttext",
                                   "infersent (2017)" = "infersent"))

résultats_bruts %>%
  filter(encoder_name != "tf-idf") %>%
  #filter(data_filename == "datapapers_clean.csv") %>%
  ggplot(aes(x=fct_reorder(encoder_name, temps, .desc=FALSE), y=temps)) +
  geom_boxplot() +
  xlab("Encodeur") +
  ylab("Temps itération")
 

résultats_bruts_mediamining <- read.csv("/home/david/Documents/Stage/sentences_embeddings/Exports/Résultats_bruts_mediamining.csv", sep=';')


names(résultats_bruts_mediamining)
unique(résultats_bruts_mediamining$data_filename)

résultats_bruts_mediamining <- résultats_bruts_mediamining %>%
  mutate(encoder_name = fct_recode(encoder_name,
                                   "sent2vec (2018)" = "sent2vec",
                                   "USE (2018)" = "USE",
                                   "fasttext (2015)" = "fasttext",
                                   "infersent (2017)" = "infersent"))

résultats_bruts_mediamining %>%
  filter(encoder_name != "tf-idf") %>%
  filter(data_filename == "experimentations.csv") %>%
  ggplot(aes(x=fct_reorder(encoder_name, temps, .desc=FALSE), y=temps)) +
  geom_boxplot() +
  xlab("Encodeur") +
  ylab("Temps itération (s)")
 