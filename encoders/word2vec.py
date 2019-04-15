import logging
# from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


# Non utilisé
# class word2vec_model(KeyedVectors):
#     def __init__(self, vec_path):
#         logger.debug("Chargement du modèle word2vec")
#         # KeyedVectors.__init__(self)
#         KeyedVectors.load_word2vec_format(vec_path)
#
#     def get_embeddings(self, text):
#         return self[text]


def word2vec_mean_model(model, data):
    mean_vectors = []
    logger.debug("fonction word2vec_mean_model")
    # Boucle sur les documents
    for doc in data:
        logger.debug(f"doc.split : {doc.split()}")
        vectors = []
        # Boucle sur les mots
        for word in doc.split():
            try:
                vectors.append(model[word].tolist())
            except Exception as e:
                logger.warning(f"{e}")
                with open("Exports/word2vec_not_found.csv", 'a+') as f:
                    f.write(f"{word}\n")
        # Calcul du vecteur moyen
        mean_vec = [float(sum(col))/len(col) for col in zip(*vectors)]
        # Ajout du vecteur moyen à la liste des vecteurs moyens
        mean_vectors.append(mean_vec)
    return mean_vectors
