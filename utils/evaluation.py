import numpy as np


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out



def fpr_evaluation(y_true, y_score, recall_level):
    """ 
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Target scores, 
        can either be probability estimates of the positive class, 
        confidence values, or non-thresholded measure of decisions
    recall_level : float
        Desired recall level
    """
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    recall = tps / tps[-1]
    thresholds = y_score[threshold_idxs]
    last_ind = tps.searchsorted(tps[-1])

    sl = slice(last_ind, None, -1)
    recall, fps, tps, thresholds = np.r_[recall[sl], 0], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
    print(thresholds)
    print(recall)
    print(fps/(np.sum(np.logical_not(y_true))))
    cutoff = np.argmin(np.abs(recall - recall_level))
    
    # Return the false positive rate at the desired recall level
    return fps[cutoff] / (np.sum(np.logical_not(y_true))) 