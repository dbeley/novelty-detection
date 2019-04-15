import numpy as np
import logging
import tensorflow as tf
import tensorflow_hub as hub
import sys


logger = logging.getLogger(__name__)

# Work around TensorFlow's absl.logging depencency which alters the
# default Python logging output behavior when present.
if 'absl.logging' in sys.modules:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    # and any other apis you want, if you want


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
