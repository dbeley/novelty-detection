import logging
from math import sqrt


logger = logging.getLogger(__name__)

def mat_conf(OBS, PRED):
    """ Indicateurs matrice de confusion """

    fp = sum([1 if obs == 0 and pred == 1 else 0 for obs, pred in zip(OBS, PRED)])
    fn = sum([1 if obs == 1 and pred == 0 else 0 for obs, pred in zip(OBS, PRED)])
    vp = sum([1 if obs == 1 and pred == 1 else 0 for obs, pred in zip(OBS, PRED)])
    vn = sum([1 if obs == 0 and pred == 0 else 0 for obs, pred in zip(OBS, PRED)])

    logger.debug(f"""
    Faux positifs : {fp}
    Faux négatifs : {fn}
    Vrais positifs : {vp}
    Vrais négatifs : {vn}""")

    return [fp, fn, vp, vn]


def all_measures(mat_conf, OBS, PRED):
    
    """ Retourne le calcul de la précison, du rappel, de l'accuracy, de la f-mesure, de la g-mean mesure """
    
    fp = mat_conf[0]
    fn = mat_conf[1]
    vp = mat_conf[2]
    vn = mat_conf[3]

    # precision
    precision = 0
    if (vp + fp) != 0:
        precision = vp / (vp + fp)

    # rappel
    rappel = 0
    if (vp + fn) != 0:
        rappel = vp / (vp + fn)

    # accuracy
    accuracy = (vp + vn) / len(OBS)

    # fscore
    b = 1
    fscore = (1 + b**2) * vp / ((1 + b**2) * vp + b**2 * fn + fp)
    if ((1 + b**2) * vp + b**2 * fn + fp) == 0.:
        fscore = 0

    # gmean
    gmean = vp / sqrt((vp + fp) * (vp + fn))
    if ((vp + fp) * (vp + fn)) == 0:
        gmean = 0

    logger.debug(f"""
    Précision : {precision}
    Rappel : {rappel}
    Accuracy : {accuracy}
    F-score : {fscore}
    G-mean : {gmean}""")

    return [precision, rappel, accuracy, fscore, gmean]
