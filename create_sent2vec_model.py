import logging
import argparse
import pandas as pd
import re
import os
from tqdm import tqdm


def main():
    args = parse_args()
    file = args.file
    print("lecture datapapers.csv")
    if file not in ["datapapers", "nytdata"]:
        logger.error(f"Fichier {file} non supporté")
        exit()
    elif not file:
        logger.error(f"Renseignez l'argument -f/--file")
        exit()
    elif file == "datapapers":
        df = pd.read_csv(os.path.expanduser("~/Documents/Données/datapapers.csv"), sep="\t", encoding="utf-8")

        list_sentences = []
        print("Traitement/extraction des phrases")

        taille_df = df.shape[0]
        for index, row in tqdm(df.iterrows(), total=taille_df):
            sentences = [x.strip() for x in re.split('\.', row['abstract'].lower())]
            list_sentences = list_sentences + sentences
        print("Export des phrases")
        with open("Exports/datapapers_sentences.csv", 'w') as f:
            for line in tqdm(list_sentences):
                if line and len(line) > 2:
                    f.write(f"{line}\n")
    elif file == "nytdata":
        df = pd.read_csv(os.path.expanduser("~/Documents/Données/data_big_category_long.csv"), sep="\t", encoding="utf-8")

        list_sentences = []
        print("Traitement/extraction des phrases")

        taille_df = df.shape[0]
        for index, row in tqdm(df.iterrows(), total=taille_df):
            sentences = [x.strip() for x in re.split('\.', row['texts'].lower())]
            list_sentences = list_sentences + sentences

        print("Export des phrases")
        with open("Exports/nytdata_sentences.csv", 'w') as f:
            for line in tqdm(list_sentences):
                if line and len(line) > 2:
                    f.write(f"{line}\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Création modèle sent2vec')
    parser.add_argument('--debug', help="Display debugging information", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)
    parser.add_argument('-f', '--file', help="File to process (datapapers ou nytdata)", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)
    return args


if __name__ == '__main__':
    main()
