import logging
import numpy as np
from gensim.models import KeyedVectors

logger = logging.getLogger(__name__)


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
        vectors = model[doc.split()]
        # vectors = [x for x in model.get_embeddings(doc.split())]
        vectors = np.array(vectors).astype(np.float)
        mean_vectors.append(vectors.mean(axis=0))
    return mean_vectors
# 
# def get_fasttext_embeddings_raw(fname, mots):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         if tokens[0] in mots:
#             logger.debug(f"{tokens[0]} trouvé")
#             yield tokens[1:]

