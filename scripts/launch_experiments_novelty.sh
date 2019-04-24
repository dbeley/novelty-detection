#!/usr/bin/env bash
# Expériences sur toutes les modalités de datapapers
# À lancer depuis la racine du projet
printf "database"
python model.py -m svm -e sent2vec -d datapapers -n database
printf "datamining"
python model.py -m svm -e sent2vec -d datapapers -n datamining
printf "medical"
python model.py -m svm -e sent2vec -d datapapers -n medical
printf "theory"
python model.py -m svm -e sent2vec -d datapapers -n theory
printf "visu"
python model.py -m svm -e sent2vec -d datapapers -n visu
printf "database fixed"
python model.py -m svm -e sent2vec -d datapapers -n database -f
printf "datamining fixed"
python model.py -m svm -e sent2vec -d datapapers -n datamining -f
printf "medical fixed"
python model.py -m svm -e sent2vec -d datapapers -n medical -f
printf "theory fixed"
python model.py -m svm -e sent2vec -d datapapers -n theory -f
printf "visu fixed"
python model.py -m svm -e sent2vec -d datapapers -n visu -f
