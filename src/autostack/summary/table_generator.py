import numpy as np
import pandas as pd
from sklearn.base import clone

def _generate_CI_table(
        base_learner_names: list[str], 
        bootstrap_idx: np.ndarray, 
        meta_features_: np.ndarray, 
        y_: np.ndarray, 
        meta_learner_) -> pd.DataFrame:
    """
    Generate summary table contains mean,
    median, 95% CI and sign stability.

    Parameters:
    ----------
    base_learner_names: list[str]
        list containint names of the base learners.

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
    stats_name =  ['mean', 'median', '2.5% quantile', '97.5% quantile', 'sign stability']

    if hasattr(meta_learner_, "intercept") and hasattr(meta_learner_,"coef"):
        row_name = ['intercept'] + base_learner_names
    elif hasattr(meta_learner_, "feature_importances_"):
        row_name = base_learner_names
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
    
    # sign stability
    signs = np.sign(bootstrap_results)
    counts = np.array([(signs == 1).sum(axis=0), (signs == -1).sum(axis=0), (signs == 0).sum(axis=0)]) / bootstrap_idx.shape[0]
    stability = np.max(counts, axis = 0)

    # form a table
    CI_table = pd.DataFrame(np.column_stack((means, median, lower, upper, stability)) ,index = row_name, columns = stats_name)

    return CI_table