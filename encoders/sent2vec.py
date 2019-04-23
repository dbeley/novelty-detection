"""
Classe sent2vec
"""

import logging
from sent2vec import Sent2vecModel

logger = logging.getLogger(__name__)


class sent2vec_model(Sent2vecModel):
    def __init__(self, model_path):
        logger.debug("Chargement du modèle sent2vec")
        Sent2vecModel.__init__(self)
        self.load_model(model_path)

    def get_embeddings(self, text):
        return self.embed_sentences(text)
