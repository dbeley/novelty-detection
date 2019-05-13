import uuid
import logging
import pandas as pd
import datetime
from evaluation.score import calcul_seuil, calcul_score
from sklearn.svm import OneClassSVM
from encoders.universal_sentence_encoder import get_USE_embeddings
from encoders.word2vec import word2vec_mean_model
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class Experiment():
    def __init__(self, encoder, samples):
        self.ID = uuid.uuid4()
        self.encoder = encoder
        self.size_historic = samples[0]
        self.size_context = samples[1]
        self.size_novelty = samples[2]
        self.heure_test = datetime.datetime.now().strftime("%Y/%m/%d %H:%M")

    def __str__(self):
        return f"Expérience {self.ID} : {self.encoder}, {self.size_historic}/{self.size_context}/{self.size_novelty}"

    def set_datasets(self, data_historic, data_context):
        self.data_historic = data_historic
        self.data_context = data_context

    def run_experiment(self, encoder_model=None):
        if self.encoder in ["infersent", "sent2vec"]:
            # classes génériques définies dans encoders/model_*.py
            # deux méthodes : __init__ et get_embeddings
            vector_historic = encoder_model.get_embeddings(list(self.data_historic.abstract.astype(str)))
            vector_context = encoder_model.get_embeddings(list(self.data_context.abstract.astype(str)))
        elif self.encoder == "USE":
            vector_historic = get_USE_embeddings(encoder_model, list(self.data_historic.abstract.astype(str)))
            vector_context = get_USE_embeddings(encoder_model, list(self.data_context.abstract.astype(str)))
        elif self.encoder == "fasttext":
            # logger.debug("Création vector_historic")
            vector_historic = word2vec_mean_model(encoder_model, list(self.data_historic.abstract.astype(str)))
            # logger.debug("Création vector_context")
            vector_context = word2vec_mean_model(encoder_model, list(self.data_context.abstract.astype(str)))
        elif self.encoder == "tf-idf":
            vectorizer = TfidfVectorizer()
            # data_all = pd.concat([self.data_historic, self.data_context], ignore_index=True)
            # X = vectorizer.fit_transform(data_all.abstract.astype(str)).toarray().tolist()
            vector_historic = vectorizer.fit_transform(self.data_historic.abstract.astype(str)).toarray().tolist()
            vector_context = vectorizer.transform(self.data_context.abstract.astype(str)).toarray().tolist()
            # vector_historic = X[0:len(self.data_historic.index)]
            # vector_context = X[len(self.data_historic.index):]
        else:
            logger.error(f"Encoder non valable, {self.encoder}")
            exit()

        self.vector_historic = vector_historic
        self.vector_context = vector_context

    def run_classif(self, method):
        if method == "score":
            # calcul du score
            logger.debug("Calcul du score")
            seuil = calcul_seuil(self.vector_historic, m='cosinus', k=2, q=0.55)
            score = calcul_score(self.vector_historic, self.vector_context, m='cosinus', k=1)
            pred = [1 if x > seuil else 0 for x in score]
        elif method == "svm":
            logger.debug("Classif avec svm")
            mod = OneClassSVM(kernel='linear', degree=3, gamma=0.5, coef0=0.5, tol=0.001, nu=0.2, shrinking=True,
                              cache_size=200, verbose=False, max_iter=-1, random_state=None)
            mod.fit(self.vector_historic)
            y_pred = mod.predict(self.vector_context)

            pred = [0 if x == 1 else 1 for x in y_pred]

            score = mod.decision_function(self.vector_context)
        else:
            logger.error(f"Problème méthode {method}. Création des embeddings impossible")
            exit()
        return score, pred
    
    def export_vectors(self):
        # Export des embeddings pour débogage
        with open(f"Exports/vector_historic_{self.encoder}_{self.ID}.csv", 'w') as f:
            for x in self.vector_historic:
                f.write(f"{x}\n")
        with open(f"Exports/vector_context_{self.encoder}_{self.ID}.csv", 'w') as f:
            for x in self.vector_context:
                f.write(f"{x}\n")
