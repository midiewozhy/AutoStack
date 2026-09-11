import numpy as np
import pandas as pd
from sklearn.base import clone

def _generate_CI_table(
        row_name: list[str], 
        stats_name: list[str],
        bootstrap_idx: np.ndarray, 
        meta_features_: np.ndarray, 
        y_: np.ndarray, 
        meta_learner_) -> pd.DataFrame:
    """
    Generate summary table contains mean,
    median, 95% CI and sign stability.

    Parameters:
    ----------
    row_name: list[str]
        list contains names of the base learners (and 'intercept').

    stats_name: list[str]
        list contains names of the statistics to calculate.

    bootstrap_idx: np.ndarray of shape [bootstrap_iter, n_samples]
        bootstrap_idx[i, j] is a row index for the j-th
        sample of the i-th bootstrapping

    meta_features_: np.ndarray of shape [n_samples, n_base_learners]
        meta feature matrix used for bootstrap and
        estimate CI.

    y_: np.ndarray of shape [n_samples,]
        Response variables.

    meta_learner_: estimator
        Meta learner.
    
    Returns:
    ----------
    CI_table: pd.DataFrame
    """

    # calculate CI, sign stability, median, mean
    bootstrap_results = np.zeros((bootstrap_idx.shape[0], len(row_name)))

    for i, idx in enumerate(bootstrap_idx):
        sample_X = meta_features_[idx, :]
        sample_y = y_[idx]
        meta = clone(meta_learner_).fit(sample_X, sample_y)
        if hasattr(meta, "intercept_") and hasattr(meta, "coef_"):
            bootstrap_results[i, 0] = meta.intercept_
            bootstrap_results[i, 1:] = meta.coef_
        elif hasattr(meta, "feature_importances_"):
            bootstrap_results[i, :] = meta.feature_importances_

    # mean median CI
    means = np.mean(bootstrap_results, axis = 0)
    median = np.median(bootstrap_results, axis = 0)
    lower = np.percentile(bootstrap_results, 2.5, axis = 0)
    upper = np.percentile(bootstrap_results, 97.5, axis = 0)
    stats = np.column_stack((means, median, lower, upper))
    
    # sign stability
    if 'intercept' in row_name:
        signs = np.sign(bootstrap_results)
        counts = np.array([(signs == 1).sum(axis=0), (signs == -1).sum(axis=0), (signs == 0).sum(axis=0)]) / bootstrap_idx.shape[0]
        stability = np.max(counts, axis = 0)
        stats = np.column_stack((stats, stability))

    # form a table
    CI_table = pd.DataFrame(stats, index = row_name, columns = stats_name)

    return CI_table