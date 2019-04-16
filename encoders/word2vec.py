import numpy as np
import logging

logger = logging.getLogger(__name__)


def word2vec_mean_model(model, data):
    mean_vectors = []
    logger.debug("fonction word2vec_mean_model")
    # Boucle sur les documents
    for doc in data:
        # logger.debug(f"doc.split : {doc.split()}")
        vectors = []
        # Boucle sur les mots
        for word in doc.split():
            try:
                vectors.append(model[str(word)].tolist())
            except Exception as e:
                # logger.warning(f"{e}")
                with open("Exports/word2vec_not_found.csv", 'a+') as f:
                    f.write(f"{word}\n")
        # Calcul du vecteur moyen
        try:
            mean_vec = [float(sum(col))/len(col) for col in zip(*vectors)]
            # logger.debug(f"mean_vec len : {len(mean_vec)}")
            # logger.debug(f"mean_vec shape : {np.array(mean_vec).shape}")
        except Exception as e:
            logger.error(f"{str(e)} - mean_vec : {mean_vec}")
        if len(mean_vec) == 0 or mean_vec is None:
            logger.warning("Moyenne incalculable, vecteur de zéros")
            mean_vec = np.zeros(shape=(300,))
        # Ajout du vecteur moyen à la liste des vecteurs moyens
        mean_vectors.append(mean_vec)
        logger.debug(f"mean_vectors shape : {np.array(mean_vectors).shape}")
    return mean_vectors
