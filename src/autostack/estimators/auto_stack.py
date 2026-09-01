import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.metrics import (
    root_mean_squared_error,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from ..core import BaseStacker
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tabulate import tabulate

class AutoStackRegressor(BaseStacker, RegressorMixin):
    """
    Standard Stacking Regressor.
    Uses OOF predictions only (no original features) as meta-features.
    """
    
    def _get_default_base_learners(self) -> list[str]:
        """
        For regression, default combination
        of base learners is:
        1. Linear Regression;
        2. Random Forest Regression;
        3. XGBoost Regression;
        4. Neural Network Regression.

        Returns:
        ----------
        base_learners: list[str]
            List containing registered
            names for default base learners.
        """
        return ['lr', 'rf', 'xgb', 'mlp']

    def _get_default_meta_learner(self) -> str:
        """
        For regression, default meta
        learner is 'ridge_cv'
        Returns:
        ----------
        meta_learner: str
            The registered name of
            default meta learner.
        """
        
        return 'ridge_cv'

    def _prepare_meta_features(self, oof_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Generate meta feature matrix.

        Parameters:
        ----------
        oof_matrix: np.ndarray of shape [n_samples, n_base_learners]
            OOF prediction matrix.
        
        X: np.ndarray of shape [n_samples, n_features]
            Predictor matrix.

        Returns:
        ----------
        meta_features: np.ndarray of shape [n_samples, n_base_learners]
            This is the same as the
            oof_matrix.
        """
        
        return oof_matrix

    def _calculate_metrics(self):
        """
        Calculate metrics for base learners
        and meta learner and return a df.

        Returns:
        ----------
        basic_table: pd.DataFrame of shape [n_base_learners+1, 4]
            basic_table[i, j] is the j-th metric value
            for the i-th learner (i=1, meta_learner)
        """
        
        # initialize metrics
        metrics = []
        learner_names = ['Meta Learner']
        columns = ['RMSE', 'MSE', 'MAE', 'R2']

        # calculate metrics for meta learner
        meta_preds = self.meta_learner_fitted_.predict(self.meta_features_)
        metrics.append(
            [root_mean_squared_error(self.y_, meta_preds),
            mean_squared_error(self.y_, meta_preds),
            mean_absolute_error(self.y_, meta_preds),
            r2_score(self.y_, meta_preds)]
        )
        
        # calculating metrics for base learners
        for i, (learner_name, _) in enumerate(self.base_learners_.items()):
            oof_preds = self.oof_matrix_[:, i]
            metrics.append(
                [root_mean_squared_error(self.y_, oof_preds),
                mean_squared_error(self.y_, oof_preds),
                mean_absolute_error(self.y_, oof_preds),
                r2_score(self.y_, oof_preds)]
            )
            learner_names.append(learner_name)

        basic_table = pd.DataFrame(metrics, index = learner_names, columns = columns)
        
        return basic_table

    def _generate_summary_table(self, summary_level: str, bootstrap_idx: np.ndarray) -> pd.DataFrame | dict:
        """
        Store statistical test results
        """
        
        if summary_level == 'basic':
            return self.basic_metrics_
        elif summary_level == 'diagnostic':
            # initialize summary dict and store basic metrics
            summary_dict = {}
            summary_dict['metrics'] = self.basic_metrics_

            # turn oof_matrix into dataframe for later calculation
            base_learner_names = list(self.base_learners_.keys())
            oof_df = pd.DataFrame(self.oof_matrix_, columns = base_learner_names)

            # calculate correlation and store correlation matrix
            cor_matrix = oof_df.corr(method = 'pearson')
            summary_dict['cor'] = cor_matrix
            
            # calculate VIF's
            vif_df = oof_df.copy()
            vif_df['intercept'] = 1

            vif_data = pd.DataFrame()
            vif_data["Feature"] = base_learner_names

            vif_data["VIF"] = [
                variance_inflation_factor(vif_df.values, i) 
                for i in range(oof_df.shape[1])
            ]

            summary_dict['vif'] = vif_data

            # calculate CI, sign stability, median, mean
            stats_name =  ['mean', 'median', '2.5% quantile', '97.5% quantile', 'sign stability']
            row_name = ['Intercept'] + base_learner_names
            bootstrap_results = np.zeros((bootstrap_idx.shape[0], len(row_name)))
            for i, idx in enumerate(bootstrap_idx):
                sample_X = self.meta_features_[idx, :]
                sample_y = self.y_[idx]
                meta = clone(self.meta_learner_).fit(sample_X, sample_y)
                bootstrap_results[i, 0] = meta.intercept_
                bootstrap_results[i, 1:] = meta.coef_

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
            summary_dict['CI'] = CI_table

            return summary_dict
            
    def _print_summary(self, summary_table: dict | pd.DataFrame, summary_level: str):
        """
        Print well-organized statistical
        tables.
        """

        if summary_level == 'basic':
            print("=" * 70)
            print("                STACK MODEL METRICS REPORT                ")
            print("=" * 70)

            print("=" * 70)
            print(tabulate(summary_table, headers ="keys", tablefmt="fancy_grid"))
            print("=" * 70)

        elif summary_level == 'diagnostic':
            print("=" * 70)
            print("                STACKING MODEL DIAGNOSTICS REPORT                ")
            print("=" * 70)
            
            # print basic metrics
            print("\n[BASIC PERFORMANCE METRICS]")
            print(tabulate(summary_table['metrics'], headers ="keys", tablefmt="fancy_grid"))
            print("=" * 70)
        
            # print bootstrap stats table
            print("\n[BOOTSTRAP COEFFICIENT ANALYSIS]")
            print(tabulate(summary_table['CI'], headers="keys", tablefmt="fancy_grid"))
            print("=" * 70)

            # print correlation matrix
            print("Correlation Matrix")
            print(tabulate(summary_table['cor'], headers="keys", tablefmt="fancy_grid"))
            print("=" * 70)
                
            # print vif data
            print("\n[MULTICOLLINEARITY (VIF)]")
            # headers="keys" automatically uses DataFrame column titles
            print(tabulate(summary_table['vif'], headers="keys", tablefmt="fancy_grid", showindex=False))
                

    def summary(self, summary_level: str = 'diagnostic', bootstrap_iter: None | int = None, return_table: bool = False) -> None | pd.DataFrame | dict:
        """
        Generate well-formatted statistical
        table based on the summary level of chosen.
        
        Parameters:
        ----------
        summary_level: str
            Determine how detailed the output
            summary is.
            Choose from {'basic', 'diagnostic',
            'full'}.
            Default to be 'diagnostic'.

        bootstrap_iter: int
            Number of times of bootstrapping
            to calculate CI.
            Default to be 100.        

        return_table: bool
            Whether to return the summary.
            Default to be False.

        Returns:
        ----------
        summary_table: None | pd.DataFrame | dict
            Table containing the summary.
        """

        # check for summary level
        if summary_level not in ['basic', 'diagnostic']:
            raise ValueError(f"Summary level can only be 'basic', or 'diagnostic', here get {summary_level}")

        # check for bootstrap_iter and doing the bootstrap again if bootstrap is not None
        if bootstrap_iter is not None:
            if isinstance(bootstrap_iter, int) and bootstrap_iter > 0:
                actual_iter = bootstrap_iter
                self.bootstrap_idx_cache_ = self._bootstrap_meta_features(self.y_.shape[0], actual_iter)
            else:
                raise ValueError('bootstrap can only be None or a positive integer.')

        # check for return_df
        if not isinstance(return_table, bool):
            raise ValueError('return_table must be a bool value.')

        # generate table
        summary_table = self._generate_summary_table(summary_level, self.bootstrap_idx_cache_)

        # print the statistical table and information
        self._print_summary(summary_table, summary_level)

        # return value
        if return_table:
            return summary_table
        