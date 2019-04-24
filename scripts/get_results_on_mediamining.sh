#!/usr/bin/env bash
# Script permettant de copier les résultats de mediamining sur la machine locale
# À lancer depuis la racine du projet
printf "Résultats bruts\n"
scp mediamining:~/Documents/sentences-embeddings/Exports/Résultats_bruts.csv Exports/Résultats_bruts_mediamining.csv
printf "Résultats condensés\n"
scp mediamining:~/Documents/sentences-embeddings/Exports/Résultats_condensés.csv Exports/Résultats_condensés_mediamining.csv
