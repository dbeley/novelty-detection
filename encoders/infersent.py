from .model_infersent import InferSent
import logging
import torch

logger = logging.getLogger(__name__)


class infersent_model(InferSent):
    def __init__(self, pkl_path, w2v_path):
        # path = "/home/david/Documents/InferSent/"
        # pkl_path = "/home/david/Documents/Données/infersent2.pkl"
        # w2v_path = "/home/david/Documents/Données/glove.840B.300d.txt"
        logger.debug("Chargement du modèle infersent")
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        InferSent.__init__(self, params_model)

        # InferSent(params_model).__init__(self, params_model)
        # model = InferSent(params_model)
        self.load_state_dict(torch.load(pkl_path))

        self.set_w2v_path(w2v_path)
        logger.debug("Construction du vocabulaire")
        self.build_vocab_k_words(K=100000)

    def get_embeddings(self, text):
        return self.encode(text, bsize=128, tokenize=False, verbose=False)
