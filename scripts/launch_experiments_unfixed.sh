#!/usr/bin/env bash
# Expériences avec échantillonnage fixe
# À lancer depuis la racine du projet
printf "launch_experiments_fixed.sh - Expériences avec échantillonnage fixe avec tous les encodeurs"
printf "1/10 - database fixed\n"
python model.py -m svm -a -d datapapers -n database
printf "2/10 - datamining fixed\n"
python model.py -m svm -a -d datapapers -n datamining
printf "3/10 - medical fixed\n"
python model.py -m svm -a -d datapapers -n medical
printf "4/10 - theory fixed\n"
python model.py -m svm -a -d datapapers -n theory
printf "5/10 - visu fixed\n"
python model.py -m svm -a -d datapapers -n visu
printf "6/10 - database fixed wihout preprocessing\n"
python model.py -m svm -a -d datapapers -n database -p
printf "7/10 - datamining fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n datamining -p
printf "8/10 - medical fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n medical -p
printf "9/10 - theory fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n theory -p
printf "10/10 - visu fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n visu -p
