#!/usr/bin/env bash
# Expériences avec échantillonnage fixe
# À lancer depuis la racine du projet
printf "launch_experiments_fixed.sh - Expériences avec échantillonnage fixe avec tous les encodeurs"
printf "1/10 - database fixed\n"
python model.py -m svm -a -d datapapers -n database -f
printf "2/10 - datamining fixed\n"
python model.py -m svm -a -d datapapers -n datamining -f
printf "3/10 - medical fixed\n"
python model.py -m svm -a -d datapapers -n medical -f
printf "4/10 - theory fixed\n"
python model.py -m svm -a -d datapapers -n theory -f
printf "5/10 - visu fixed\n"
python model.py -m svm -a -d datapapers -n visu -f 
printf "6/10 - database fixed wihout preprocessing\n"
python model.py -m svm -a -d datapapers -n database -f -p
printf "7/10 - datamining fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n datamining -f -p
printf "8/10 - medical fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n medical -f -p
printf "9/10 - theory fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n theory -f -p
printf "10/10 - visu fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n visu -f -p
