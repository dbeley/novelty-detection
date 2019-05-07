import random
import logging

logger = logging.getLogger(__name__)


def split_data(data, size_historic, size_context, size_novelty, theme, fix_seed):
    """ Fonction qui genere le contexte et l historique """
    if fix_seed:
        random.seed(9583)
    novelty = data[data.theme == str(theme)]
    no_novelty = data[data.theme != str(theme)]
    idx_novelty = list(novelty.index)
    idx_no_novelty = list(no_novelty.index)

    idx_all = random.sample(idx_no_novelty, size_historic + size_context - size_novelty)
    # historique -> taille 0:size_historic (2000)
    idx_historic = idx_all[0:size_historic]
    # contexte -> taille sizehistoric: + 20 dans idx_novelty
    idx_context = idx_all[size_historic:] + random.sample(idx_novelty, size_novelty)
    data_historic = data.iloc[idx_historic]
    data_context = data.iloc[idx_context]

    return data_historic, data_context
