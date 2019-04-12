import pandas as pd
import numpy as np
import random
import argparse
import logging
import time
import sys
import os
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from encoders.infersent import infersent_model
from encoders.sent2vec import sent2vec_model
from evaluation.score import calcul_seuil, calcul_score
from evaluation.measures import mat_conf, all_measures

pd.np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger()
temps_debut = time.time()

# Variables globales
PATH_INFERSENT_PKL = os.path.expanduser("~/Documents/Données/infersent2.pkl")
PATH_INFERSENT_W2V = os.path.expanduser("~/Documents/Données/glove.840B.300d.txt")
PATH_SENT2VEC_BIN = os.path.expanduser("~/Documents/Données/torontobooks_unigrams.bin")
SUPPORTED_ENCODER = ["infersent", "USE", "sent2vec"]
SUPPORTED_METHOD = ["score", "svm"]
ITERATION_NB = 1
#       historic, context, novelty
SAMPLES_LIST = [[2000, 300, 50],
                # [2000, 300, 10],
                # [2000, 300, 150],
                # [2000, 500, 10],
                # [2000, 500, 50],
                # [2000, 500, 100],
                # [2000, 500, 250],
                # [5000, 500, 100],
                # [5000, 500, 20]
                ]


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
    try:
        data = pd.read_csv('Exports/datapapers_clean.csv')
        data.columns = ['id', 'abstract', 'theme']
        data = data.drop(['id'], axis=1)
    except Exception as e:
        logger.error(str(e))
        logger.error("Fichier datapapers_clean.csv non trouvé dans le dossier Exports. Lancez le script prepare.py")
        exit()

    # parsage des arguments
    theme = args.novelty

    all_encoders = args.all_encoders
    encoder = [args.encoder]
    if encoder[0] not in SUPPORTED_ENCODER and all_encoders is None:
        logger.error(f"encoder {encoder} non implémenté. Choix = {SUPPORTED_ENCODER}")
        exit()
    elif all_encoders:
        logger.debug("option all_encoders sélectionnée. Sélection de tous les encodeurs")
        encoder = SUPPORTED_ENCODER

    method = args.method
    if method not in SUPPORTED_METHOD:
        logger.error(f"méthode {method} non implémentée. Choix = {SUPPORTED_METHOD}")
        exit()

    # à décommenter pour créer l'entête
    variables_list = ['theme',
                      'encoder_name',
                      'methode',
                      'novelty class',
                      'size_historic',
                      'size_context',
                      'size_novelty',
                      'iteration',
                      'AUC',
                      'temps',
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
        f.write(f"{';'.join(variables_list)}\n")

    import tensorflow as tf
    import tensorflow_hub as hub

    # Boucle sur les encodeurs sélectionnés
    for single_encoder in encoder:
        print(f"Novelty = {theme}, encodeur = {single_encoder}, méthode = {method}")

        # Chargement de l'encodeur
        logger.debug("Chargement de l'encoder")
        if single_encoder == "infersent":
            encoder_model = infersent_model(pkl_path=PATH_INFERSENT_PKL, w2v_path=PATH_INFERSENT_W2V)
        elif single_encoder == "sent2vec":
            encoder_model = sent2vec_model(model_path=PATH_SENT2VEC_BIN)
        elif single_encoder == "USE":
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

            # Import the Universal Sentence Encoder's TF Hub module
            encoder_model = hub.Module(module_url)

        # Boucle sur les paramètres d'échantillons définis dans samples_list
        for exp in SAMPLES_LIST:
            size_historic = exp[0]
            size_context = exp[1]
            size_novelty = exp[2]
            # Boucle d'itération
            for iteration in range(0, ITERATION_NB):
                iteration_begin = time.time()
                print(f"iteration : {iteration}, size_historic : {size_historic}, size_context : {size_context}, size_novelty : {size_novelty}")
                data_historic, data_context = split_data(data, size_historic=size_historic, size_context=size_context, size_novelty=size_novelty, theme=theme)

                # Export des jeux de données pour débogage
                # data_historic.to_csv('Exports/datapapers_historic.csv')
                # data_context.to_csv('Exports/datapapers_context.csv')

                logger.debug("Création des embeddings")
                if single_encoder in ["infersent", "sent2vec"]:
                    # classes génériques définies dans encoders/model_*.py
                    # deux méthodes : __init__ et get_embeddings
                    vector_historic = encoder_model.get_embeddings(list(data_historic.abstract.astype(str)))
                    vector_context = encoder_model.get_embeddings(list(data_context.abstract.astype(str)))
                elif single_encoder == "USE":
                    # classe spécifique pour USE
                    with tf.Session() as session:
                        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
                        vector_historic = np.array(session.run(encoder_model(list(data_historic.abstract.astype(str))))).tolist()
                        vector_context = np.array(session.run(encoder_model(list(data_context.abstract.astype(str))))).tolist()
                        session.close()

                # classification
                if method == "score":
                    # calcul du score
                    logger.debug("Calcul du score")
                    seuil = calcul_seuil(vector_historic, m='cosinus', k=2, q=0.55)
                    score = calcul_score(vector_historic, vector_context, m='cosinus', k=1)
                    pred = [1 if x > seuil else 0 for x in score]
                elif method == "svm":
                    logger.debug("Classif avec svm")
                    mod = OneClassSVM(kernel='linear', degree=3, gamma=0.5, coef0=0.5, tol=0.001, nu=0.2, shrinking=True,
                                      cache_size=200, verbose=False, max_iter=-1, random_state=None)
                    mod.fit(vector_historic)
                    y_pred = mod.predict(vector_context)
                    pred = [0 if x == 1 else 1 for x in y_pred]

                    score = mod.decision_function(vector_context)
                else:
                    logger.error(f"Problème méthode {method}. Création des embeddings impossible")
                    exit()

                obs = [1 if elt == theme else 0 for elt in data_context.theme]
                obs2 = [-1 if x == 1 else 1 for x in obs]

                matrice_confusion = mat_conf(obs, pred)
                mesures = all_measures(obs, pred)
                logger.debug(matrice_confusion)
                logger.debug(mesures)

                AUC = roc_auc_score(obs2, score)

                logger.debug(AUC)
                iteration_time = "%.2f" % (time.time() - iteration_begin)

                # Arrondi des résultats numériques avant export
                AUC = round(AUC, 2)
                # iteration_time = round(iteration_time, 2)
                # matrice_confusion = [round(x, 2) for x in matrice_confusion]
                mesures = [round(x, 2) for x in mesures]

                # Export des résultats
                with open(f"Exports/Résultats.csv", 'a+') as f:
                    f.write(f"{ theme };{ single_encoder };{ method };{theme};{ size_historic };{ size_context };{ size_novelty };{ iteration+1 };{ AUC };{iteration_time};{';'.join(map(str, matrice_confusion))};{';'.join(map(str, mesures))}\n")

    logger.debug("Runtime : %.2f seconds" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Script principal')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-m', '--method', help="Méthode (score ou svm)", type=str)
    parser.add_argument('-e', '--encoder', help="Encodeur (infersent ou USE/universal sentence encoder", type=str)
    parser.add_argument('-a', '--all_encoders', help="Active tous les encodeurs implémentés", dest='all_encoders', action='store_true')
    parser.add_argument('-n', '--novelty', help="Nouveauté à découvrir (défaut = 'theory')", type=str, default='theory')
    parser.set_defaults(all_encoders=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
