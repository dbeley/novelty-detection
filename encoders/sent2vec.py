from sent2vec import Sent2vecModel
import logging

logger = logging.getLogger(__name__)


class sent2vec_model(Sent2vecModel):
    def __init__(self):
        model = "/home/david/Documents/Données/torontobooks_unigrams.bin"
        logger.debug("Chargement du modèle sent2vec")
        Sent2vecModel.__init__(self)
        self.load_model(model)

    def get_embeddings(self, text):
        return self.embed_sentences(text)
