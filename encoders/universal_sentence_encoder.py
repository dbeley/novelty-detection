"""
Classe USE (Universal Sentence Encoder)
"""

import numpy as np
import logging
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)


class hub_module(hub.Module):
    def __init__(self, url):
        logger.debug("Création modèle hub Universal Sentence Encoding")
        hub.Module.__init__(self, url)


def get_USE_embeddings(model, messages):
    with tf.Session() as session:
        session.run(
            [tf.global_variables_initializer(), tf.tables_initializer()]
        )
        message_embeddings = np.array(
            session.run(model(list(messages)))
        ).tolist()
        session.close()
    return message_embeddings
