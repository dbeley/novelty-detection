import logging
import numpy as np
from gensim.models import KeyedVectors
from statistics import mean

logger = logging.getLogger(__name__)


# Non utilisé
class word2vec_model(KeyedVectors):
    def __init__(self, vec_path):
        logger.debug("Chargement du modèle word2vec")
        # KeyedVectors.__init__(self)
        KeyedVectors.load_word2vec_format(vec_path)

    def get_embeddings(self, text):
        return self[text]


def word2vec_mean_model(model, data):
    mean_vectors = []
    logger.debug("fonction word2vec_mean_model")
    for doc in data:
        logger.debug(f"doc.split : {doc.split()}")
        # vectors = model.get_embeddings(doc.split())
        vectors = []
        for word in doc.split():
            try:
                vectors.append(model[word].tolist())
            except Exception as e:
                logger.warning(f"{e}")
                with open("Exports/word2vec_not_found.csv", 'a+') as f:
                    f.write(f"{word}\n")
        # vectors = [x for x in model.get_embeddings(doc.split())]
        # vectors = np.array(vectors).astype(np.float)
        mean_vec = [float(sum(col))/len(col) for col in zip(*vectors)]
        mean_vectors.append(mean_vec)
        # mean_vectors.append(vectors.mean(axis=0))
    return mean_vectors
