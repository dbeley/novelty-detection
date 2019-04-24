#!/usr/bin/env bash
# Expériences sur toutes les modalités de datapapers
# À lancer depuis la racine du projet
printf "launch_experiments_novelty.sh - Expériences sur toutes les modalités de datapapers avec sent2vec"
printf "1/10 - database\n"
python model.py -m svm -e sent2vec -d datapapers -n database
printf "2/10 - datamining\n"
python model.py -m svm -e sent2vec -d datapapers -n datamining
printf "3/10 - medical\n"
python model.py -m svm -e sent2vec -d datapapers -n medical
printf "4/10 - theory\n"
python model.py -m svm -e sent2vec -d datapapers -n theory
printf "5/10 - visu\n"
python model.py -m svm -e sent2vec -d datapapers -n visu
printf "6/10 - database fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n database -f
printf "7/10 - datamining fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n datamining -f
printf "8/10 - medical fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n medical -f
printf "9/10 - theory fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n theory -f
printf "10/10 - visu fixed\n"
python model.py -m svm -e sent2vec -d datapapers -n visu -f
