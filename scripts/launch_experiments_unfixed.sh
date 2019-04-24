#!/usr/bin/env bash
# Expériences avec échantillonnage fixe
# À lancer depuis la racine du projet
printf "launch_experiments_unfixed.sh - Expériences avec tous les encodeurs\n"
printf "1/10 - database\n"
python model.py -m svm -a -d datapapers -n database
printf "2/10 - datamining\n"
python model.py -m svm -a -d datapapers -n datamining
printf "3/10 - medical\n"
python model.py -m svm -a -d datapapers -n medical
printf "4/10 - theory\n"
python model.py -m svm -a -d datapapers -n theory
printf "5/10 - visu\n"
python model.py -m svm -a -d datapapers -n visu
printf "6/10 - database wihout preprocessing\n"
python model.py -m svm -a -d datapapers -n database -p
printf "7/10 - datamining without preprocessing\n"
python model.py -m svm -a -d datapapers -n datamining -p
printf "8/10 - medical without preprocessing\n"
python model.py -m svm -a -d datapapers -n medical -p
printf "9/10 - theory without preprocessing\n"
python model.py -m svm -a -d datapapers -n theory -p
printf "10/10 - visu without preprocessing\n"
python model.py -m svm -a -d datapapers -n visu -p
