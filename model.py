import pandas as pd
import numpy as np
import random
import argparse
import logging
import time
import sys
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from encoders.infersent import infersent_model
from evaluation.score import calcul_seuil, calcul_score
from evaluation.measures import mat_conf, all_measures

pd.np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger()
temps_debut = time.time()

# Variables globales
SUPPORTED_ENCODER = ["infersent", "USE"]
SUPPORTED_METHOD = ["score", "svm"]
NB_ITERATION = 50


def split_data(data, size_historic, size_context, size_novelty, theme):
    """ Fonction qui genere le contexte et l historique """
    novelty = data[data.theme == str(theme)]
    no_novelty = data[data.theme != str(theme)]
    idx_novelty = list(novelty.index)
    idx_no_novelty = list(no_novelty.index)

    idx_all = random.sample(idx_no_novelty, size_historic + size_context - size_novelty)
    # data_historic =  taille 0:size_historic (2000)
    # data_context = taille sizehistoric: + 20 dans idx_novelty

    idx_historic = idx_all[0:size_historic]
    idx_context = idx_all[size_historic:] + random.sample(idx_novelty, size_novelty)
    data_historic = data.iloc[idx_historic]
    data_context = data.iloc[idx_context]

    return data_historic, data_context


def main():
    args = parse_args()

    # chargement des données
    data = pd.read_csv('Exports/datapapers_clean.csv')
    data.columns = ['id', 'abstract', 'theme']
    data = data.drop(['id'], axis=1)

    # parsage des arguments
    # theme = args.novelty
    theme = "theory"

    encoder_name = args.encoder
    if encoder_name not in SUPPORTED_ENCODER:
        logger.error(f"encoder {encoder_name} non implémenté. Choix = {supported_encoder_name}")
        exit()

    method = args.method
    if method not in SUPPORTED_METHOD:
        logger.error(f"méthode {method} non implémentée. Choix = {supported_method}")
        exit()

    # à décommenter pour créer l'entête
    liste_variables = ['theme',
                       'encoder_name',
                       'methode',
                       'size_historic',
                       'size_context',
                       'size_novelty',
                       'iteration',
                       'AUC',
                       'temps'
                       'faux positifs',
                       'faux négatifs',
                       'vrais positifs',
                       'vrais négatifs',
                       'précision',
                       'rappel',
                       'accuracy',
                       'fscore',
                       'gmean'
                       ]
    with open(f"Exports/Résultats.csv", 'a+') as f:
        # f.write(f"theme;encoder_name;methode;size_historic;size_context;size_novelty;iteration;faux positifs;faux négatifs;vrais positifs;vrais négatifs;AUC;temps\n")
        f.write(f"{';'.join(liste_variables)}\n")

    #       historic, context, novelty
    liste_exp = [[2000, 300, 50],
                 [2000, 500, 50],
                 [2000, 500, 100],
                 [2000, 500, 200],
                 [3000, 300, 20],
                 [3000, 500, 200],
                 [3000, 1000, 500]]

    import tensorflow as tf
    import tensorflow_hub as hub

    logger.debug("Chargement de l'encoder")
    # encoder = load_infersent_model()
    if encoder_name == "infersent":
        encoder = infersent_model()
    elif encoder_name == "USE":
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

        # Import the Universal Sentence Encoder's TF Hub module
        encoder = hub.Module(module_url)
        # encoder = USE_model()

    # Tests
    print(f"Novelty = {theme}, modèle = {encoder_name}, méthode = {method}")
    for exp in liste_exp:
        size_historic = exp[0]
        size_context = exp[1]
        size_novelty = exp[2]
        for iteration in range(0, NB_ITERATION):
            temps_debut_iteration = time.time()
            print(f"iteration : {iteration}, size_historic : {size_historic}, size_context : {size_context}, size_novelty : {size_novelty}")

            data_historic, data_context = split_data(data, size_historic=size_historic, size_context=size_context, size_novelty=size_novelty, theme=theme)

            # data_historic.to_csv('Exports/datapapers_historic.csv')
            # data_context.to_csv('Exports/datapapers_context.csv')

            # encoder
            logger.debug("Création des embeddings")
            if encoder_name == "infersent":
                vector_historic = encoder.get_embeddings(list(data_historic.abstract.astype(str)))
                # vector_historic = encoder.encode(list(data_historic.abstract.astype(str)), bsize=128, tokenize=False, verbose=False)
                vector_context = encoder.get_embeddings(list(data_context.abstract.astype(str)))
                # vector_context = encoder.encode(list(data_context.abstract.astype(str)), bsize=128, tokenize=False, verbose=False)
            elif encoder_name == "USE":
                with tf.Session() as session:
                    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                    vector_historic = np.array(session.run(encoder(list(data_historic.abstract.astype(str))))).tolist()
                    vector_context = np.array(session.run(encoder(list(data_context.abstract.astype(str))))).tolist()

            # classification
            if method == "score":
                # calcul du score
                logger.debug("Calcul du score")
                seuil = calcul_seuil(vector_historic, m='cosinus', k=2, q=0.55)
                score = calcul_score(vector_historic, vector_context, m='cosinus', k=1)
                pred = [1 if x > seuil else 0 for x in score]
            elif method == "svm":
                logger.debug("Calcul avec svm")
                mod = OneClassSVM(kernel='linear', degree=3, gamma=0.5, coef0=0.5, tol=0.001, nu=0.2, shrinking=True,
                                  cache_size=200, verbose=False, max_iter=-1, random_state=None)
                mod.fit(vector_historic)
                y_pred = mod.predict(vector_context)
                pred = [0 if x == 1 else 1 for x in y_pred]

                score = mod.decision_function(vector_context)


            obs = [1 if elt == theme else 0 for elt in data_context.theme]
            obs2 = [-1 if x == 1 else 1 for x in obs]

            matrice_confusion = mat_conf(obs, pred)
            mesures = all_measures(obs, pred)
            logger.debug(matrice_confusion)
            logger.debug(mesures)

            AUC = roc_auc_score(obs2, score)

            logger.debug(AUC)
            temps_iteration = "%.2f" % (time.time() - temps_debut_iteration)

            # Export des résultats
            with open(f"Exports/Résultats.csv", 'a+') as f:
                f.write(f"{ theme };{ encoder_name };{ method };{ size_historic };{ size_context };{ size_novelty };{ iteration+1 };{ AUC };{temps_iteration};{';'.join(map(str, matrice_confusion))};{';'.join(map(str, mesures))}\n")

    logger.debug("Runtime : %.2f seconds" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Script principal')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-m', '--method', help="Méthode (score ou svm)", type=str)
    parser.add_argument('-e', '--encoder', help="Encodeur (infersent ou USE/universal sentence encoder", type=str)
    parser.add_argument('-n', '--novelty', help="Nouveauté à découvrir (défaut = 'theory')", type=str)
    # parser.add_argument('-t', '--train', help="Train", dest='train', action='store_true')
    # parser.set_defaults(train=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
