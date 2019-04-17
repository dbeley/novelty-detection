import numpy as np
import logging

logger = logging.getLogger(__name__)


def word2vec_mean_model(model, data):
    logger.debug("Chargement du modèle word2vec_mean_model")
    mean_vectors = []
    # Boucle sur les documents
    for doc in data:
        vectors = []
        # Boucle sur les mots
        for word in doc.split():
            try:
                vectors.append(model[str(word)].tolist())
            except Exception as e:
                with open("Exports/word2vec_not_found.csv", 'a+') as f:
                    f.write(f"{word}\n")
        # Calcul du vecteur moyen
        try:
            mean_vec = [float(sum(col))/len(col) for col in zip(*vectors)]
        except Exception as e:
            logger.error(f"{str(e)} - mean_vec : {mean_vec}")
        if len(mean_vec) == 0 or mean_vec is None:
            logger.warning("Moyenne incalculable, vecteur de zéros")
            mean_vec = np.zeros(shape=(300,))
        # Ajout du vecteur moyen à la liste des vecteurs moyens
        mean_vectors.append(mean_vec)
    return mean_vectors
