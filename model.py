"""
Script principal.
Effectue les tests renseignés par les variables globales et les arguments entrés lors du lancement du script.
"""

import datetime
import pandas as pd
import numpy as np
import random
import argparse
import logging
import time
import sys
import os
import uuid
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from encoders.infersent import infersent_model
from encoders.sent2vec import sent2vec_model
from encoders.word2vec import word2vec_mean_model
from evaluation.score import calcul_seuil, calcul_score
from evaluation.measures import mat_conf, all_measures
from encoders.universal_sentence_encoder import hub_module, get_USE_embeddings
from gensim.models import KeyedVectors

pd.np.set_printoptions(threshold=sys.maxsize)
logger = logging.getLogger()

temps_debut = time.time()

# Variables globales
# infersent
PATH_INFERSENT_PKL = os.path.expanduser("~/Documents/Données/infersent2.pkl")
PATH_INFERSENT_W2V = os.path.expanduser("~/Documents/Données/glove.840B.300d.txt")
# sent2vec
# PATH_SENT2VEC_BIN = os.path.expanduser("~/Documents/Données/torontobooks_unigrams.bin")
PATH_SENT2VEC_BIN = os.path.expanduser("~/Documents/Données/datapapers_model.bin")
# fasttext
# PATH_FASTTEXT = os.path.expanduser("~/Documents/Données/crawl-300d-2M.vec")
PATH_FASTTEXT = os.path.expanduser("~/Documents/Données/wiki-news-300d-1M.vec")
# PATH_FASTTEXT = os.path.expanduser("~/Documents/Données/datapapers_fasttext_model.vec")
SUPPORTED_DATASETS = ['datapapers', 'nytdata']
PATH_DATAPAPERS = os.path.expanduser("~/Documents/Données/datapapers.csv")
PATH_NYTDATA = os.path.expanduser("~/Documents/Données/data_big_category_long.csv")
SUPPORTED_ENCODERS = ["sent2vec", "fasttext", "USE", "infersent"]
SUPPORTED_METHODS = ["score", "svm"]
ITERATION_NB = 50
#       historic, context, novelty
SAMPLES_LIST = [[2000, 300, 50],
                [2000, 300, 150],
                [2000, 300, 280],
                [5000, 300, 50],
                [5000, 300, 150],
                [5000, 300, 280],
                [2000, 1000, 50],
                [2000, 1000, 100],
                [2000, 1000, 250],
                [2000, 1000, 500],
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

    # Work around TensorFlow's absl.logging depencency which alters the
    # default Python logging output behavior when present.
    if 'absl.logging' in sys.modules:
        logger.info("Tentative de correction de la verbosité des logs")
        import absl.logging
        if args.loglevel == 20:
            logger.info("Logging : info")
            absl.logging.set_verbosity('info')
            absl.logging.set_stderrthreshold('info')
        if args.loglevel == 10:
            logger.info("Logging : debug")
            absl.logging.set_verbosity('debug')
            absl.logging.set_stderrthreshold('debug')
        # and any other apis you want, if you want

    # parsage des arguments
    dataset = args.dataset
    without_preprocessing = args.without_preprocessing
    theme = args.novelty

    # chargement des données

    if dataset == 'datapapers':
        if without_preprocessing:
            logger.debug("Utilisation du jeu de données datapapers.csv")
            data_filename = "datapapers.csv"
            try:
                data = pd.read_csv(PATH_DATAPAPERS, sep="\t", encoding="utf-8")
                data = data.drop(['id', 'conf', 'title', 'author', 'year', 'eq', 'conf_short'], axis=1)
            except Exception as e:
                logger.error(str(e))
                logger.error(f"Fichier {data_filename} non trouvé.")
                exit()
        else:
            logger.debug("Utilisation du jeu de données datapapers_clean.csv")
            data_filename = "datapapers_clean.csv"
            try:
                data = pd.read_csv(f'Exports/{data_filename}')
                data.columns = ['id', 'abstract', 'theme']
                data = data.drop(['id'], axis=1)
            except Exception as e:
                logger.error(str(e))
                logger.error(f"Fichier {data_filename} non trouvé. Lancez le script prepare.py.")
                exit()
    elif dataset == 'nytdata':
        if without_preprocessing:
            logger.debug("Utilisation du jeu de données data_big_category_long.csv")
            data_filename = "nytdata.csv"
            try:
                data = pd.read_csv(PATH_NYTDATA, sep="\t", encoding="utf-8")
                data = data.drop(['week', 'titles'], axis=1)
                data.rename(columns={'texts': 'abstract', 'principal_classifier': 'theme', 'second_classifier': 'theme2', 'third_classifier': 'theme3'}, inplace=True)
            except Exception as e:
                logger.error(str(e))
                logger.error(f"Fichier {data_filename} non trouvé.")
                exit()
        else:
            logger.debug("Utilisation du jeu de données nytdata_clean.csv")
            data_filename = "nytdata_clean.csv"
            try:
                data = pd.read_csv(f'Exports/{data_filename}')
                # data = data.drop(['week', 'second_classifier', 'titles', 'third_classifier'], axis=1)
                data.rename(columns={'texts': 'abstract', 'principal_classifier': 'theme', 'second_classifier': 'theme2', 'third_classifier': 'theme3'}, inplace=True)
            except Exception as e:
                logger.error(str(e))
                logger.error(f"Fichier {data_filename} non trouvé. Lancez le script prepare.py.")
                exit()
    elif dataset is None:
        logger.error(f"Entrez un jeu de données avec l'argument -d/--dataset parmi {SUPPORTED_DATASETS}.")
        exit()
    else:
        logger.error(f"Jeu de données {dataset} non supporté. Jeux de données supportés : {SUPPORTED_DATASET}")
        exit()


    all_encoders = args.all_encoders
    encoder = [args.encoder]

    if encoder[0] in SUPPORTED_ENCODERS and not all_encoders:
        logger.debug(f"Encodeur {encoder[0]} sélectionné.")
    elif all_encoders:
        logger.debug("Option all_encoders sélectionnée. Sélection de tous les encodeurs")
        encoder = SUPPORTED_ENCODERS
    else:
        logger.error("Utiliser -e ou -a pour sélectionner un ou plusieurs encodeurs. -h pour plus d'informations.")
        exit()

    method = args.method
    if method not in SUPPORTED_METHODS:
        logger.error(f"Méthode {method} non implémentée. Choix = {SUPPORTED_METHODS}")
        exit()

    variables_list = ['ID',
                      'data_filename',
                      'date test',
                      'theme',
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
    raw_results_filename = "Exports/Résultats_bruts.csv"
    condensed_results_filename = "Exports/Résultats_condensés.csv"

    # si le fichier n'existe pas, on le crée et y insère l'entête
    if not os.path.isfile(raw_results_filename):
        logger.debug(f"Création du fichier {raw_results_filename}")
        with open(raw_results_filename, 'a+') as f:
            f.write(f"{';'.join(variables_list)}\n")

    if not os.path.isfile(condensed_results_filename):
        logger.debug(f"Création du fichier {condensed_results_filename}")
        with open(condensed_results_filename, 'a+') as f:
            f.write(f"{';'.join(variables_list)}\n")

    # Boucle sur les encodeurs sélectionnés
    for single_encoder in encoder:

        # Chargement de l'encodeur
        logger.debug("Chargement de l'encoder")
        if single_encoder == "infersent":
            encoder_model = infersent_model(pkl_path=PATH_INFERSENT_PKL, w2v_path=PATH_INFERSENT_W2V)
        elif single_encoder == "sent2vec":
            encoder_model = sent2vec_model(model_path=PATH_SENT2VEC_BIN)
        elif single_encoder == "USE":
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

            # # Import the Universal Sentence Encoder's TF Hub module
            encoder_model = hub_module(module_url)
        elif single_encoder == "fasttext":
            logger.info(f"Chargement du modèle fasstext ({PATH_FASTTEXT}) (~15 minutes)")
            encoder_model = KeyedVectors.load_word2vec_format(PATH_FASTTEXT)

        # Boucle sur les paramètres d'échantillons définis dans samples_list
        for exp in SAMPLES_LIST:
            # Récupération des paramètres d'échantillons
            size_historic = exp[0]
            size_context = exp[1]
            size_novelty = exp[2]

            # ID du test
            ID_test = uuid.uuid4()
            # Heure du test
            heure_test = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")

            # Liste des résultats
            AUC_list = []
            matrice_confusion_list = []
            mesures_list = []
            iteration_time_list = []

            # Résumé du test
            logger.info(f"Dataset = {data_filename}, Novelty = {theme}, encodeur = {single_encoder}, méthode = {method}, size_historic : {size_historic}, size_context : {size_context}, size_novelty : {size_novelty}")

            # Boucle d'itération
            for iteration in tqdm(range(0, ITERATION_NB)):
                iteration_begin = time.time()
                logger.debug(f"iteration : {iteration}")
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
                    vector_historic = get_USE_embeddings(encoder_model, list(data_historic.abstract.astype(str)))
                    vector_context = get_USE_embeddings(encoder_model, list(data_context.abstract.astype(str)))
                elif single_encoder == "fasttext":
                    # logger.debug("Création vector_historic")
                    vector_historic = word2vec_mean_model(encoder_model, list(data_historic.abstract.astype(str)))
                    # logger.debug("Création vector_context")
                    vector_context = word2vec_mean_model(encoder_model, list(data_context.abstract.astype(str)))

                # Export des embeddings pour débogage
                # with open(f"Exports/vector_historic_{single_encoder}.csv", 'w') as f:
                #     for x in vector_historic:
                #         f.write(f"{x}\n")
                # with open(f"Exports/vector_context_{single_encoder}.csv", 'w') as f:
                #     for x in vector_context:
                #         f.write(f"{x}\n")

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
                logger.debug(f"matrice : {matrice_confusion}")
                mesures = all_measures(matrice_confusion, obs, pred)

                AUC = roc_auc_score(obs2, score)

                logger.debug(f"AUC : {AUC}")
                iteration_time = "%.2f" % (time.time() - iteration_begin)
                logger.debug(f"temps itération = {iteration_time}")

                # Ajout des résultats dans une liste
                AUC_list.append(AUC)
                matrice_confusion_list.append(matrice_confusion)
                mesures_list.append(mesures)
                iteration_time_list.append(float(iteration_time))

                # Arrondi des résultats numériques avant export
                AUC = round(AUC, 2)
                # iteration_time = round(iteration_time, 2)
                # matrice_confusion = [round(x, 2) for x in matrice_confusion]
                mesures = [round(x, 2) for x in mesures]

                # Export des résultats bruts
                logger.debug("Exports des résultats bruts")
                with open(raw_results_filename, 'a+') as f:
                    f.write(f"{ID_test};{data_filename};{heure_test};{theme};{single_encoder};{method};{theme};{size_historic};{size_context};{size_novelty};{iteration+1};{AUC};{iteration_time};{';'.join(map(str, matrice_confusion))};{';'.join(map(str, mesures))}\n")

            # Création résultats condensés
            AUC_condensed = round(sum(AUC_list) / float(len(AUC_list)), 2)
            iteration_time_condensed = round(sum(iteration_time_list) / float(len(iteration_time_list)), 2)
            matrice_confusion_condensed = np.round(np.mean(np.array(matrice_confusion_list), axis=0), 2)
            mesures_condensed = np.round(np.mean(np.array(mesures_list), axis=0), 2)

            # Export des résultats condensés
            logger.debug("Exports des résultats condensés")
            with open(condensed_results_filename, 'a+') as f:
                f.write(f"{ID_test};{data_filename};{heure_test};{theme};{single_encoder};{method};{theme};{size_historic};{size_context};{size_novelty};{iteration+1};{AUC_condensed};{iteration_time_condensed};{';'.join(map(str, matrice_confusion_condensed))};{';'.join(map(str, mesures_condensed))}\n")

    logger.info("Temps d'exécution : %.2f secondes" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Script principal')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-m', '--method', help="Méthode (score ou svm)", type=str)
    parser.add_argument('-e', '--encoder', help="Encodeur mots/phrases/documents (infersent, sent2vec, USE ou fasttext)", type=str)
    parser.add_argument('-a', '--all_encoders', help="Active tous les encodeurs implémentés", dest='all_encoders', action='store_true')
    parser.add_argument('-p', '--without_preprocessing', help="Utilise le jeu de données sélectionné, mais sans pré-traitement (phrases complètes)", dest='without_preprocessing', action='store_true')
    parser.add_argument('-d', '--dataset', help="Jeu de données à utiliser (datapapers ou nytdata)", type=str)
    parser.add_argument('-n', '--novelty', help="Nouveauté à découvrir (défaut = 'theory')", type=str, default='theory')
    parser.set_defaults(all_encoders=False, without_preprocessing=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
