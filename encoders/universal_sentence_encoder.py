import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import logging

logger = logging.getLogger(__name__)
# logger = logging.getLogger('tensorflow')
# logger.setLevel(logging.INFO)


# class USE_model(tf.Session):
#     def __init__(self):
#         logger.debug("Création du modèle USE")
#         # InferSent.__init__(self, params_model)
#         tf.Session.__init__(self)
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
# 
#         self.embed = hub.Module(module_url)
#         self.run([tf.global_variables_initializer(), tf.tables_initializer()])
#         logger.debug("Création du modèle USE terminée")
# 
#     def get_embeddings(self, messages):
#         return self.run(self.embed(messages))
# 
# 
# class USE_model(hub.Module):
#     def __init__(self):
#         logger.debug("Création du modèle USE")
#         module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#         hub.Module.__init__(self, module_url)
#         logger.debug("Création du modèle USE terminée")
# 
#     def get_embeddings(self, messages):
#         with tf.Session() as session:
#             session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#             message_embeddings = session.run(self(messages))
#             return session.run(self(messages))


class hub_module(hub.Module):
    def __init__(self, url):
        logger.debug("Création modèle hub")
        hub.Module.__init__(self, url)

def get_USE_embeddings(model, messages):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = np.array(session.run(model(list(messages)))).tolist()
        session.close()
    return message_embeddings
