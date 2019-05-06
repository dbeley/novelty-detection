"""
Script principal.
Effectue les tests renseignés par les variables globales et les arguments entrés lors du lancement du script.
Version utilisation les jeux de données du dossier Exports/datapapers_fixed
"""

import pandas as pd
import numpy as np
import argparse
import logging
import time
import sys
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from encoders.infersent import infersent_model
from encoders.sent2vec import sent2vec_model
from evaluation.measures import mat_conf, all_measures
from encoders.universal_sentence_encoder import hub_module
from gensim.models import KeyedVectors
from experiments.experiment import Experiment
from experiments.data_functions import split_data

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

PATH_DATAPAPERS_FIXED = os.path.expanduser("~/Documents/Stage/sentences_embeddings/Exports/datapapers_fixed")
SUPPORTED_ENCODERS = ["sent2vec", "fasttext", "USE", "infersent", "tf-idf"]
# SUPPORTED_ENCODERS = ["sent2vec", "USE", "infersent"]
SUPPORTED_METHODS = ["score", "svm"]
ITERATION_NB = 50
#       historic, context, novelty
SAMPLES_LIST = [[2000, 300, 5],
                [2000, 300, 10],
                [2000, 300, 20],
                ]


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

    # Parsage des arguments
    theme = 'theory'
    fix_seed = False

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

    # Variables pour les résultats bruts
    raw_variables_list = ['ID',
                          'fixed_sample',
                          'data_filename',
                          'date test',
                          'theme',
                          'encoder_name',
                          'model_name',
                          'methode',
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

    # Variables pour les résultats condensés
    condensed_variables_list = ['ID',
                                'fixed_sample',
                                'data_filename',
                                'date test',
                                'theme',
                                'encoder_name',
                                'model_name',
                                'methode',
                                'size_historic',
                                'size_context',
                                'size_novelty',
                                'iteration',
                                'AUC',
                                'temps',
                                'moy. faux positifs',
                                'moy. faux négatifs',
                                'moy. vrais positifs',
                                'moy. vrais négatifs',
                                'moy. précision',
                                'moy. rappel',
                                'moy. accuracy',
                                'moy. fscore',
                                'moy. gmean',
                                'std. faux positifs',
                                'std. faux négatifs',
                                'std. vrais positifs',
                                'std. vrais négatifs',
                                'std. précision',
                                'std. rappel',
                                'std. accuracy',
                                'std. fscore',
                                'std. gmean',
                                'q0.25 faux positifs',
                                'q0.25 faux négatifs',
                                'q0.25 vrais positifs',
                                'q0.25 vrais négatifs',
                                'q0.25 précision',
                                'q0.25 rappel',
                                'q0.25 accuracy',
                                'q0.25 fscore',
                                'q0.25 gmean',
                                'q0.5 faux positifs',
                                'q0.5 faux négatifs',
                                'q0.5 vrais positifs',
                                'q0.5 vrais négatifs',
                                'q0.5 précision',
                                'q0.5 rappel',
                                'q0.5 accuracy',
                                'q0.5 fscore',
                                'q0.5 gmean',
                                'q0.75 faux positifs',
                                'q0.75 faux négatifs',
                                'q0.75 vrais positifs',
                                'q0.75 vrais négatifs',
                                'q0.75 précision',
                                'q0.75 rappel',
                                'q0.75 accuracy',
                                'q0.75 fscore',
                                'q0.75 gmean'
                                ]

    raw_results_filename = "Exports/Résultats_bruts.csv"
    condensed_results_filename = "Exports/Résultats_condensés.csv"

    # si le fichier n'existe pas, on le crée et y insère l'entête
    if not os.path.isfile(raw_results_filename):
        logger.debug(f"Création du fichier {raw_results_filename}")
        with open(raw_results_filename, 'a+') as f:
            f.write(f"{';'.join(raw_variables_list)}\n")

    if not os.path.isfile(condensed_results_filename):
        logger.debug(f"Création du fichier {condensed_results_filename}")
        with open(condensed_results_filename, 'a+') as f:
            f.write(f"{';'.join(condensed_variables_list)}\n")

    # Boucle sur les encodeurs sélectionnés
    for single_encoder in encoder:

        # Chargement de l'encodeur
        logger.debug("Chargement de l'encodeur")

        # Initialisation de l'encodeur
        model_name = "Non applicable"
        if single_encoder == "infersent":
            encoder_model = infersent_model(pkl_path=PATH_INFERSENT_PKL, w2v_path=PATH_INFERSENT_W2V)
            model_name = PATH_INFERSENT_W2V
        elif single_encoder == "sent2vec":
            encoder_model = sent2vec_model(model_path=PATH_SENT2VEC_BIN)
            model_name = PATH_SENT2VEC_BIN
        elif single_encoder == "USE":
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

            # Import the Universal Sentence Encoder's TF Hub module
            encoder_model = hub_module(module_url)
        elif single_encoder == "fasttext":
            logger.info(f"Chargement du modèle fasttext ({PATH_FASTTEXT}) (~15 minutes)")
            encoder_model = KeyedVectors.load_word2vec_format(PATH_FASTTEXT)
            model_name = PATH_FASTTEXT
        elif single_encoder == "tf-idf":
            logger.info("Utilisation de TF-IDF")
            encoder_model = None

        # Mise en forme du nom du modèle
        model_name = Path(model_name).stem

        for exp in SAMPLES_LIST:
            # Générateur fichiers (10 normalement)
            experiment = Experiment(single_encoder, exp)

            # Chargement des données datapapers_fixed pour cette expérience
            gen_files = sorted(Path(PATH_DATAPAPERS_FIXED).glob(f"**/context_{experiment.size_historic}_{experiment.size_context}_{experiment.size_novelty}_*"))

            # Liste des résultats
            AUC_list = []
            matrice_confusion_list = []
            mesures_list = []
            iteration_time_list = []

            # Résumé du test
            logger.info(f"Paramètres : Données = datapapers_fixed, Nouveauté = {theme}, Encodeur = {experiment.encoder}, Méthode = {method}, Historique/Contexte/Nouveauté : {experiment.size_historic}/{experiment.size_context}/{experiment.size_novelty}")

            # Boucle sur les 10 fichiers
            for iteration in gen_files:
                iteration_begin = time.time()

                data_context = pd.read_csv(iteration, sep='\t')
                # n_data_historic = str(PATH_DATAPAPERS_FIXED) + "/historic_" + '_'.join(i.stem.split('/')[-1].split('_')[1:]) + ".csv"
                n_data_historic = "Exports/datapapers_fixed/historic_" + '_'.join(iteration.stem.split('/')[-1].split('_')[1:]) + ".csv"
                data_filename = n_data_historic
                print(n_data_historic)
                data_historic = pd.read_csv(n_data_historic, sep='\t')
                data_historic.columns = ['id', 'abstract', 'theme']
                data_context.columns = ['id', 'abstract', 'theme']
                # Assignation de l'historique et du context
                experiment.set_datasets(data_historic, data_context)

                # Application de l'encodeur, création des vecteurs d'embeddings
                experiment.run_experiment(encoder_model)

                # Exports des vecteurs pour débogage
                # experiment.export_vectors()

                # Application de la classification
                try:
                    score, pred = experiment.run_classif(method)
                except Exception as e:
                    logger.error(e)
                    exit()

                obs = [1 if elt == theme else 0 for elt in experiment.data_context.theme]
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
                    f.write(f"{experiment.ID};{fix_seed};{data_filename};{experiment.heure_test};{theme};{experiment.encoder};{model_name};{method};{experiment.size_historic};{experiment.size_context};{experiment.size_novelty};{iteration};{AUC};{iteration_time};{';'.join(map(str, matrice_confusion))};{';'.join(map(str, mesures))}\n")

            # Création résultats condensés
            AUC_condensed = round(sum(AUC_list) / float(len(AUC_list)), 2)
            # iteration_time_condensed = round(sum(iteration_time_list) / float(len(iteration_time_list)), 2)
            iteration_time_condensed = 'rien'
            mean_matrice_confusion_condensed = np.round(np.mean(np.array(matrice_confusion_list), axis=0), 2)
            mean_mesures_condensed = np.round(np.mean(np.array(mesures_list), axis=0), 2)
            std_matrice_confusion_condensed = np.round(np.std(np.array(matrice_confusion_list), axis=0), 2)
            std_mesures_condensed = np.round(np.std(np.array(mesures_list), axis=0), 2)
            quantile025_matrice_confusion_condensed = np.round(np.quantile(np.array(matrice_confusion_list), 0.25, axis=0), 2)
            quantile025_mesures_condensed = np.round(np.quantile(np.array(mesures_list), 0.25, axis=0), 2)
            med_matrice_confusion_condensed = np.round(np.quantile(np.array(matrice_confusion_list), 0.5, axis=0), 2)
            med_mesures_condensed = np.round(np.quantile(np.array(mesures_list), 0.5, axis=0), 2)
            quantile075_matrice_confusion_condensed = np.round(np.quantile(np.array(matrice_confusion_list), 0.75, axis=0), 2)
            quantile075_mesures_condensed = np.round(np.quantile(np.array(mesures_list), 0.75, axis=0), 2)

            # Export des résultats condensés
            logger.debug("Exports des résultats condensés")
            with open(condensed_results_filename, 'a+') as f:
                f.write(f"{experiment.ID};{fix_seed};{data_filename};{experiment.heure_test};{theme};{experiment.encoder};{model_name};{method};{experiment.size_historic};{experiment.size_context};{experiment.size_novelty};{iteration};{AUC_condensed};{iteration_time_condensed};{';'.join(map(str, mean_matrice_confusion_condensed))};{';'.join(map(str, mean_mesures_condensed))};{';'.join(map(str, std_matrice_confusion_condensed))};{';'.join(map(str, std_mesures_condensed))};{';'.join(map(str, quantile025_matrice_confusion_condensed))};{';'.join(map(str, quantile025_mesures_condensed))};{';'.join(map(str, med_matrice_confusion_condensed))};{';'.join(map(str, med_mesures_condensed))};{';'.join(map(str, quantile075_matrice_confusion_condensed))};{';'.join(map(str, quantile075_mesures_condensed))}\n")

    logger.info("Temps d'exécution : %.2f secondes" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description='Script principal')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-m', '--method', help="Méthode (score ou svm)", type=str)
    parser.add_argument('-e', '--encoder', help="Encodeur mots/phrases/documents (infersent, sent2vec, USE, fasttext ou tf-idf)", type=str)
    parser.add_argument('-a', '--all_encoders', help="Active tous les encodeurs implémentés", dest='all_encoders', action='store_true')
    parser.set_defaults(all_encoders=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
