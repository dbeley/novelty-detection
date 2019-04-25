# Fasttext

## Création de la liste des phrases

```
python create_sent2vec_model.py
```

## Création du modèle

```
./fasttext skipgram -input /home/david/Documents/sentences_embeddings/Exports/datapapers_sentences.csv -output datapapers_fasttext_model
```

# Sent2vec

## Création du modèle

```
./fasttext sent2vec -input /home/david/Documents/sentences_embeddings/Exports/datapapers_sentences.csv -output datapapers_sent2vec_model -minCount 5 -dim 700 -epoch 5 -thread 4
```
