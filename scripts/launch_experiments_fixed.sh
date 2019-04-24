#!/usr/bin/env bash
# Expériences avec échantillonnage fixe
# À lancer depuis la racine du projet
printf "database fixed\n"
python model.py -m svm -a -d datapapers -n database -f
printf "datamining fixed\n"
python model.py -m svm -a -d datapapers -n datamining -f
printf "medical fixed\n"
python model.py -m svm -a -d datapapers -n medical -f
printf "theory fixed\n"
python model.py -m svm -a -d datapapers -n theory -f
printf "visu fixed\n"
python model.py -m svm -a -d datapapers -n visu -f 
printf "database fixed wihout preprocessing\n"
python model.py -m svm -a -d datapapers -n database -f -p
printf "datamining fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n datamining -f -p
printf "medical fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n medical -f -p
printf "theory fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n theory -f -p
printf "visu fixed without preprocessing\n"
python model.py -m svm -a -d datapapers -n visu -f -p
