#!/usr/bin/env bash
# Expériences avec nytdata
# À lancer depuis la racine du projet
printf "launch_experiments_nytdata.sh - Expériences avec nytdata pour sent2vec"
printf "1/10 - football\n"
python model.py -m svm -e sent2vec -d nytdata -n football
printf "2/10 - finances\n"
python model.py -m svm -e sent2vec -d nytdata -n finances
printf "3/10 - music\n"
python model.py -m svm -e sent2vec -d nytdata -n music
printf "4/10 - theater\n"
python model.py -m svm -e sent2vec -d nytdata -n theater
printf "5/10 - elections\n"
python model.py -m svm -e sent2vec -d nytdata -n elections
printf "6/10 - housing\n"
python model.py -m svm -e sent2vec -d nytdata -n housing
printf "7/10 - motion pictures\n"
python model.py -m svm -e sent2vec -d nytdata -n motion pictures
printf "8/10 - art\n"
python model.py -m svm -e sent2vec -d nytdata -n art
printf "9/10 - terrorism\n"
python model.py -m svm -e sent2vec -d nytdata -n terrorism
printf "10/10 - united states international relations\n"
python model.py -m svm -e sent2vec -d nytdata -n united states international relations