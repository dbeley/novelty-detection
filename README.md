# sentences_embeddings

## Requirements

- python-nltk
- python-gensim
- python-pytorch
- sent2vec
- infersent

## Usage

### Préparation des données

Variable globale :

- DATAPAPERS : localisation du jeu de données datapapers.csv

#### Exemple d'utilisation

```
python prepare.py -h
python prepare.py
```

### Script principal

Variables globales :

- PATH_INFERSENT_PKL = localisation du modèle infersent
- PATH_INFERSENT_W2V = localisation vecteurs de mots
- PATH_SENT2VEC_BIN = localisation du modèle sent2vec
- SUPPORTED_ENCODER = liste des encodeurs supportés ["infersent", "USE", "sent2vec"]
- SUPPORTED_METHOD = liste des classifieurs supportés ["score", "svm"]
- ITERATION_NB = nombre d'itérations
- SAMPLES_LIST = liste d'échantillonages de la forme [taille historique, taille contexte, taille nouveauté (inclus dans contexte)]

```
python model.py -h
python model.py
```
