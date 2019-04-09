import pandas as pd
import numpy as np
import random
import argparse
import logging
import time
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

logger = logging.getLogger()
temps_debut = time.time()
pd.np.set_printoptions(threshold=sys.maxsize)


def load_infersent_model(path="/home/david/Documents/InferSent/", pkl_path="/home/david/Documents/Données/infersent2.pkl", w2v_path = "/home/david/Documents/Données/glove.840B.300d.txt"):
    """ Construit le model infersent """
    # os.chdir(path)

    logger.debug("Chargement du modèle infersent")
    from models import InferSent
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
    model = InferSent(params_model)
    model.load_state_dict(torch.load(pkl_path))

    model.set_w2v_path(w2v_path)
    logger.debug("Construction du vocabulaire")
    model.build_vocab_k_words(K=100000)

    return model


def split_data(DATA, size_historic, size_context, size_novelty, theme):
    """ Fonction qui genere le contexte et l historique """
    novelty = DATA[DATA.theme == str(theme)]
    no_novelty = DATA[DATA.theme != str(theme)]
    idx_novelty = list(novelty.index)
    idx_no_novelty = list(no_novelty.index)

    idx_all = random.sample(idx_no_novelty, size_historic + size_context - size_novelty)
    # data_historic =  taille 0:size_historic (2000)
    # data_context = taille sizehistoric: + 20 dans idx_novelty

    idx_historic = idx_all[0:size_historic]
    idx_context = idx_all[size_historic:] + random.sample(idx_novelty, size_novelty)
    DATA_historic = DATA.iloc[idx_historic]
    DATA_context = DATA.iloc[idx_context]

    return DATA_historic, DATA_context


def score_novelty(vector_historic, vector_context, m='cosinus', k=1, s=0):

    """ Score de nouveauté """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - vector_context: les vecteurs représentatifs des documents du contexte (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier > 0)
                  - s: un pas de décalage pour considérer le (s+1)eme à (s+k+1)eme plus proche voisin (entier >= 0)
    """

    """ Sorties :
                  Renvoie un vecteur de score pour chaque document du contexte testé (np.array)
    """

    if m.lower() == 'pearson':
        vector_context = np.transpose(np.transpose(vector_context) - vector_context.mean(axis=1))
        vector_historic = np.transpose(np.transpose(vector_historic) - vector_historic.mean(axis=1))
        
    c = abs(cosine_similarity(vector_context, vector_historic))
    d = 1 - c
    d.sort(axis=1)

    if k < 1:
        k = 1

    if k < s:
        k = s + 1

    return np.sum(d[:, s:(k+s)], axis=1) / k


def calcul_seuil(vector_historic, m='cosinus', k=1, q=0.95):

    """ Determination du seuil """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier strictement positif)
                  - q: ordre du quantile d'ordre q (pourcentage)
    """

    """ Sorties :
                  Renvoie un seuil pour la detection de la nouveauté (float)
    """

    res = score_novelty(vector_historic, vector_historic, m=m, k=k, s=1)
    return pd.Series(res).quantile(q)


def calcul_score(vector_historic, vector_context, m='cosinus', k=1, s=0):

    """ Score de nouveauté """

    """ Entrées :
                  - vector_historic: les vecteurs représentatifs des documents de l'historique (np.array)
                  - vector_context: les vecteurs représentatifs des documents du contexte (np.array)
                  - m: la mesure utilisée, Cosinus ou Pearson (string)
                  - k: le nombre de plus proches voisins utilisés (entier > 0)
                  - s: un pas de décalage pour considérer le (s+1)eme à (s+k+1)eme plus proche voisin (entier >= 0)
    """

    """ Sorties :
                  Renvoie un vecteur de score pour chaque document du contexte testé (np.array)
    """

    if m.lower() == 'pearson':
        vector_context = np.transpose(np.transpose(vector_context) - vector_context.mean(axis = 1))
        vector_historic = np.transpose(np.transpose(vector_historic) - vector_historic.mean(axis = 1))

    c = abs(cosine_similarity(vector_context, vector_historic))
    d = 1 - c
    d.sort(axis=1)

    if k < 1:
        k = 1

    if k < s:
        k = s + 1

    return (np.sum(d[:, s:(k+s)], axis=1) / k)


def mat_conf(OBS, PRED):
    fp = sum([1 if obs == 0 and pred == 1 else 0 for obs, pred in zip(OBS, PRED)])
    fn = sum([1 if obs == 1 and pred == 0 else 0 for obs, pred in zip(OBS, PRED)])
    vp = sum([1 if obs == 1 and pred == 1 else 0 for obs, pred in zip(OBS, PRED)])
    vn = sum([1 if obs == 0 and pred == 0 else 0 for obs, pred in zip(OBS, PRED)])

    return fp, fn, vp, vn


def main():
    args = parse_args()
    data = pd.read_csv('Exports/datapapers_clean.csv')
    data.columns = ['id', 'abstract', 'theme']
    data = data.drop(['id'], axis=1)

    theme = "theory"
    modele = "infersent"
    # methode = "score"
    methode = args.method
    if methode not in ["score", "svm"]:
        logger.error(f"méthode {methode} non implémentée. Sortie du programme.")
        exit()
    nb_iteration = 5

    # à décommenter pour créer l'entête
    with open(f"Exports/Résultats.csv", 'a+') as f:
        f.write(f"theme,modele,methode,size_historic,size_context,size_novelty,iteration,faux positifs,faux négatifs,vrais positifs,vrais négatifs,AUC,temps\n")

    # logger.debug(data)
    list_size_historic = [1000, 1000, 1000, 1000, 1000]
    list_size_context = [300, 300, 300, 100, 100]
    list_size_novelty = [200, 100, 20, 50, 20]
    # list_size_historic = [2000, 2000, 2000, 4000, 4000, 4000, 500, 500, 500]
    # list_size_context = [300, 300, 300, 500, 1000, 1000, 30, 300]
    # list_size_novelty = [20, 100, 200, 300, 100, 500, 10, 100]

    model = load_infersent_model()

    for size_historic, size_context, size_novelty in zip(list_size_historic, list_size_context, list_size_novelty):
        for iteration in range(0, nb_iteration):
            temps_debut_iteration = time.time()
            logger.debug(f"Nouvelle boucle, iteration {iteration}")
            logger.debug(f"size_historic : {size_historic}, size_context : {size_context}, size_novelty : {size_novelty}")

            # data_historic, data_context = split_data(data, size_historic=2000, size_context=300, size_novelty=20, theme=theme)
            data_historic, data_context = split_data(data, size_historic, size_context, size_novelty, theme=theme)
            data_historic.to_csv('Exports/datapapers_historic.csv')
            data_context.to_csv('Exports/datapapers_context.csv')

            # modèle infersent
            # model = load_infersent_model()
            logger.debug("Création des embeddings")
            vector_historic = model.encode(list(data_historic.abstract.astype(str)), bsize=128, tokenize=False, verbose=False)
            vector_context = model.encode(list(data_context.abstract.astype(str)), bsize=128, tokenize=False, verbose=False)

            with open('Exports/vector_historic.txt', 'w') as f:
                for item in vector_historic:
                    f.write("%s\n" % item)

            with open('Exports/vector_context.txt', 'w') as f:
                for item in vector_context:
                    f.write("%s\n" % item)

            if methode == "score":
                # calcul du score
                logger.debug("Calcul du score")
                seuil = calcul_seuil(vector_historic, m='cosinus', k=2, q=0.55)
                # logger.debug(f"seuil : {seuil}")
                score = calcul_score(vector_historic, vector_context, m='cosinus', k=1)
                # logger.debug(f"score : {score}")
                pred = [1 if x > seuil else 0 for x in score]
                # logger.debug(pred)
            if methode == "svm":
                logger.debug("Calcul avec svm")
                mod = OneClassSVM(kernel='linear', degree=3, gamma=0.5, coef0=0.5, tol=0.001, nu=0.2, shrinking=True,
                                  cache_size=200, verbose=False, max_iter=-1, random_state=None)
                mod.fit(vector_historic)
                y_pred = mod.predict(vector_context)
                pred = [0 if x == 1 else 1 for x in y_pred]

                score = mod.decision_function(vector_context)

            logger.debug(f"Novelty = {theme}, modèle = {modele}, méthode = {methode}")
            # affiche_mat_conf(OBS, PRED)

            obs = [1 if elt == theme else 0 for elt in data_context.theme]
            obs2 = [-1 if x == 1 else 1 for x in obs]

            fp, fn, vp, vn = mat_conf(obs, pred)

            AUC = roc_auc_score(obs2, score)

            logger.debug(AUC)
            # Conserve_resultats['{}'.format(methode)] += np.array(all_measures(OBS, PRED) + [AUC])
            temps_iteration = "%.2f" % (time.time() - temps_debut_iteration)
            with open(f"Exports/Résultats.csv", 'a+') as f:
                f.write(f"{ theme },{ modele },{ methode },{ size_historic },{ size_context },{ size_novelty },{ iteration+1 },{fp},{fn},{vp},{vn},{ AUC },{temps_iteration}\n")

    logger.debug("Runtime : %.2f seconds" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Preparation')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-m', '--method', help="Méthode", type=str)
    # parser.add_argument('-t', '--train', help="Train", dest='train', action='store_true')
    # parser.set_defaults(train=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
