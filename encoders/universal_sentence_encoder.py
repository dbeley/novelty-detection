import tensorflow as tf
import tensorflow_hub as hub
import logging

logger = logging.getLogger(__name__)


class USE_model(tf.Session):
    def __init__(self):
        logger.debug("Création du modèle USE")
        # InferSent.__init__(self, params_model)
        tf.Session.__init__(self)
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

        self.embed = hub.Module(module_url)
        self.run([tf.global_variables_initializer(), tf.tables_initializer()])
        logger.debug("Création du modèle USE terminée")

    def get_embeddings(self, messages):
        return self.run(self.embed(messages))


class USE_model(hub.Module):
    def __init__(self):
        logger.debug("Création du modèle USE")
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
        hub.Module.__init__(self, module_url)
        logger.debug("Création du modèle USE terminée")

    def get_embeddings(self, messages):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(self(messages))
            return session.run(self(messages))
