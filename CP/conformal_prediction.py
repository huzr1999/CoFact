import numpy as np
from typing import List

def score_func(
    claim_scores : List[np.ndarray],
    annotations : List[np.ndarray],
    method : str = "max"
):
    if method == "max":
        min_score = -1
        scores = np.zeros((len(claim_scores),))
        for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
            scores[i] = np.max(cs[~a]) if np.sum(~a) >= 1 else min_score
    if isinstance(method, int):
        min_score = -1
        scores = np.zeros((len(claim_scores),))
        for i, (cs, a) in enumerate(zip(claim_scores, annotations)):
            try: 
                scores[i] = np.sort(cs[~a])[::-1][method] if np.sum(~a) > method else min_score
            except:
                import IPython; IPython.embed()
    return scores


# def split_dataset(dataset, rng, train_frac):
#     score, x, y, rep = dataset
#     ind = np.arange(len(x))
#     rng.shuffle(ind)
#     train_num = int(train_frac * len(x))
#     train_ind = ind[0:train_num]
#     calib_ind = ind[train_num:]

#     score_train = [score[i] for i in train_ind]
#     x_train = x[train_ind]
#     y_train = [y[i] for i in train_ind]
#     rep_train = [rep[i] for i in train_ind]

#     score_calib = [score[i] for i in calib_ind]
#     x_calib = x[calib_ind]
#     y_calib = [y[i] for i in calib_ind]
#     rep_calib = [rep[i] for i in calib_ind]

#     return [score_train, x_train, y_train, rep_train], [score_calib, x_calib, y_calib, rep_calib]



def get_retained_claims(claim_scores, thresholds):
    claims_retained = []
    for cs, t in zip(claim_scores, thresholds):
        claims_retained.append(np.mean(cs > t))
    return claims_retained

def get_retained_claim_indices(claim_scores, thresholds):
    claims_retained = []
    for cs, t in zip(claim_scores, thresholds):
        claims_retained.append(np.where(cs > t)[0])
    return claims_retained

def get_validity(claim_scores, annotations, threshold, method):
    conf_scores = score_func(claim_scores, annotations, method)
    validity = conf_scores <= threshold
    return validity

def split_threshold(
    conf_scores : np.ndarray,
    quantile
):
    n = len(conf_scores)
    threshold = np.sort(conf_scores)[int(np.ceil(quantile * (n + 1)))]
    return threshold

def run_split_conformal(x_arr, y_arr, method, quantile):
    conf_scores = score_func(x_arr, y_arr, method=method)
    threshold = split_threshold(conf_scores, quantile)
    return conf_scores, threshold