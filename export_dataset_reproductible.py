"""
Script d'export des jeux de données
"""

import pandas as pd
import random
import argparse
import logging
import time
import os
import errno

logger = logging.getLogger()

temps_debut = time.time()

# Variables globales
PATH_DATAPAPERS = os.path.expanduser("~/Documents/Données/datapapers.csv")
PATH_DATAPAPERS_CLEAN = os.path.expanduser("~/Documents/Stage/sentences_embeddings/Exports/datapapers_clean.csv")

# historic, context, novelty
SAMPLES_LIST = [[2000, 300, 5],
                [2000, 300, 10],
                [2000, 300, 20],
                # [2000, 300, 0],
                # [5000, 20, 20],
                # [5000, 0, 0]
                # [2000, 300, 5],
                # [2000, 300, 10],
                # [2000, 300, 50]
                # [2000, 300, 50],
                # [2000, 300, 150],
                # [2000, 300, 280],
                # [5000, 300, 50],
                # [5000, 300, 150],
                # [5000, 300, 280],
                # [2000, 1000, 50],
                # [2000, 1000, 100],
                # [2000, 1000, 250],
                # [2000, 1000, 500],
                ]
# seed à utiliser pour l'échantillonnage
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def split_data(data, size_historic, size_context, size_novelty, theme, fix_seed=9583):
    """ Fonction qui genere le contexte et l historique """
    random.seed(fix_seed)

    novelty = data[data.theme == str(theme)]
    no_novelty = data[data.theme != str(theme)]
    idx_novelty = list(novelty.index)
    idx_no_novelty = list(no_novelty.index)

    idx_all = random.sample(idx_no_novelty, size_historic + size_context - size_novelty)
    idx_historic = idx_all[0:size_historic]
    idx_context = idx_all[size_historic:] + random.sample(idx_novelty, size_novelty)
    data_historic = data.iloc[idx_historic]
    data_context = data.iloc[idx_context]

    return data_historic, data_context


def main():
    args = parse_args()

    # Parsage des arguments
    without_preprocessing = args.without_preprocessing
    theme = args.novelty

    # Création du dossier Exports si non existant
    try:
        os.makedirs("Exports/datapapers_fixed")
    except FileExistsError:
        # directory already exists
        logger.info("Dossier d'export déjà existant.")
        pass

    # Chargement du jeu de données
    if without_preprocessing:
        logger.info("Utilisation du jeu de données datapapers.csv")
        data_filename = "datapapers.csv"
        try:
            data = pd.read_csv(PATH_DATAPAPERS, sep="\t", encoding="utf-8")
            data = data.drop(['id', 'conf', 'title', 'author', 'year', 'eq', 'conf_short'], axis=1)
        except Exception as e:
            logger.error(str(e))
            logger.error(f"Fichier {data_filename} non trouvé.")
            exit()
    else:
        logger.info("Utilisation du jeu de données datapapers_clean.csv")
        data_filename = "datapapers_clean.csv"
        try:
            data = pd.read_csv(PATH_DATAPAPERS_CLEAN)
            data.columns = ['id', 'abstract', 'theme']
            data = data.drop(['id'], axis=1)
        except Exception as e:
            logger.error(str(e))
            logger.error(f"Fichier {data_filename} non trouvé. Lancez le script prepare.py.")
            exit()

    # Boucle sur les seeds et les paramètres d'échantillons
    for seed in SEEDS:
        logger.info(f"Seed : {seed}")
        for exp in SAMPLES_LIST:
            # Récupération des paramètres d'échantillons
            size_historic = exp[0]
            size_context = exp[1]
            size_novelty = exp[2]
            logger.info(f"historique : {size_historic}, contexte : {size_context}, nouveauté : {size_novelty}")

            data_historic, data_context = split_data(data, size_historic=size_historic, size_context=size_context, size_novelty=size_novelty, theme=theme, fix_seed=seed)
            if without_preprocessing:
                data_historic.to_csv(f"Exports/datapapers_fixed/historic_{size_historic}_{size_context}_{size_novelty}_s{seed}_{theme}_datapapers.csv", sep='\t')
                data_context.to_csv(f"Exports/datapapers_fixed/context_{size_historic}_{size_context}_{size_novelty}_s{seed}_{theme}_datapapers.csv", sep='\t')
            else:
                data_historic.to_csv(f"Exports/datapapers_fixed/historic_{size_historic}_{size_context}_{size_novelty}_s{seed}_{theme}_datapapers_clean.csv", sep='\t')
                data_context.to_csv(f"Exports/datapapers_fixed/context_{size_historic}_{size_context}_{size_novelty}_s{seed}_{theme}_datapapers_clean.csv", sep='\t')

    logger.info("Temps d'exécution : %.2f secondes" % (time.time() - temps_debut))


def parse_args():
    parser = argparse.ArgumentParser(description="Script d'export des jeux de données découpés en historique/contexte")
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-p', '--without_preprocessing', help="Utilise le jeu de données sélectionné, mais sans pré-traitement (phrases complètes)", dest='without_preprocessing', action='store_true')
    parser.add_argument('-n', '--novelty', help="Nouveauté à découvrir (défaut = 'theory')", type=str, default='theory')
    parser.set_defaults(without_preprocessing=False)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
