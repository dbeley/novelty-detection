#!/usr/bin/env bash
# Expériences sur toutes les modalités de datapapers
# À lancer depuis la racine du projet
printf "database\n"
python model.py -m svm -e sent2vec -d datapapers -n database
printf "datamining\n"
python model.py -m svm -e sent2vec -d datapapers -n datamining
printf "medical\n"
python model.py -m svm -e sent2vec -d datapapers -n medical
printf "theory\n"
python model.py -m svm -e sent2vec -d datapapers -n theory
printf "visu\n"
python model.py -m svm -e sent2vec -d datapapers -n visu
printf "database fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n database -f
printf "datamining fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n datamining -f
printf "medical fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n medical -f
printf "theory fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n theory -f
printf "visu fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n visu -f
