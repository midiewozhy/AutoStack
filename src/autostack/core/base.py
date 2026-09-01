from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from ..registry import BASE_LEARNER_REGISTRY, META_LEARNER_REGISTRY

class BaseStacker(ABC, BaseEstimator):
    """
    This class is the father class of every stacking classes.
    Responsible for generating K-fold OOF matrix, parameter tuning,
    and parallel training.

    Parameters:
    ----------
    base_learners: str | list[str] | dict[str, any]
        The base learners for stacking.
        Default to be 'auto' specified by type of mission.

    meta_learner: str
        The meta learner for stacking.
        Default to be 'auto' specified by type of mission.

    kf: int
        The number of folds used to generate
        oof_matrix.
        Default to be 5.

    error_handling: str
        How to deal with error during base learner training.
        Default to be 'warn'

    verbose: int
        Verbosity.
        Default to be 1

    bootstrap_iter: int
        Number of times of bootstrapping
        to calculate CI.
        Default to be 100.

    random_state: int
        Random seed.
        Default to be 42.
        
    Attributes:
    ----------
    base_learners_: dict[str, estimator]
        Contain the base learners
        without fitting.

    meta_learner_: estimator
        Meta learner without
        fitting.

    full_models_: dict[str, estimator]
        Contain the fitted base
        learners.

    meta_features_: np.ndarray
        Design matrix for meta learner.
    """
    
    def __init__(
        self,
        base_learners: str | list[str] | dict[str, any]  = 'auto',
        meta_learner: str = 'auto',
        kf: int = 5,
        error_handling: str = 'warn',
        verbose: int = 1,
        bootstrap_iter: int = 100,
        random_state: int = 42
    ):

        # check parameter type
        # check for base learners
        if isinstance(base_learners, str):
            if base_learners != 'auto':
                raise ValueError(f"base_learners string must be 'auto'. Got '{base_learners}'.")
        elif isinstance(base_learners, list):
            for i, item in enumerate(base_learners):
                if not isinstance(item, str):
                    raise TypeError(f"List items must be strings (registry keys). Item at index {i} has type {type(item).__name__}.")
                if item not in BASE_LEARNER_REGISTRY:
                    raise ValueError(f"Base learner '{item}' is not registered. Available keys: {list(BASE_LEARNER_REGISTRY.keys())}")
        elif isinstance(base_learners, dict):
            if not all(isinstance(k, str) for k in base_learners.keys()):
                invalid_keys = [type(k).__name__ for k in base_learners.keys() if not isinstance(k, str)]
                raise TypeError(f"All dictionary keys must be strings. Found key types: {set(invalid_keys)}")
            for key, value in base_learners.items():
                if hasattr(value, 'fit') and hasattr(value, 'predict'):
                    pass
                else:
                    raise TypeError(
                        f"Dictionary value for key '{key}' must be a string (registry key) or an estimator object with 'fit' and 'predict' methods."
                        f"Got {type(value).__name__}."
                    )
        else:
            raise TypeError(
                f"base_learners must be 'auto', a list of strings, or a dict mapping names to estimator objects. Got {type(base_learners).__name__}."
            )

        # check for meta learner
        if isinstance(meta_learner, str):
            if meta_learner != 'auto' and meta_learner not in META_LEARNER_REGISTRY:
                raise ValueError(f"meta_learner must be 'auto' or one of {list(META_LEARNER_REGISTRY.keys())}, got '{meta_learner}'.")
        elif hasattr(meta_learner, 'fit') and hasattr(meta_learner, 'predict'):
            pass
        else:
            raise TypeError(f"meta_learner must be a string or an estimator object with 'fit' and 'predict' methods. Got {type(meta_learner).__name__}.")

        # check for cv
        if not isinstance(kf, int) or kf < 1:
            raise TypeError(f"kf must be an integer, found {type(kf).__name__}.")
            
        # check for error handling
        if error_handling not in ['raise', 'warn', 'ignore']:
            raise ValueError(f"Error handling type can only be 'raise', 'warn'  or 'ignore', here get '{error_handling}'")

        #check for verbosity
        if verbose not in [0, 1, 2]:
            raise ValueError(f"Verbosity levels are from 0 to 2, here get {verbose}")

        # check for num of bootstrap
        if not isinstance(bootstrap_iter, int) or bootstrap_iter < 1:
            raise ValueError("bootstrap_iter must be a positive integer.")

        # check for random state
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer.")
        
        self.base_learners = base_learners
        self.meta_learner = meta_learner
        self.kf = kf
        self.error_handling = error_handling
        self.verbose = verbose
        self.bootstrap_iter = bootstrap_iter
        self.random_state = random_state

    @abstractmethod
    def _get_default_base_learners(self) -> list[str]:
        """
        Get the default list of base learners for regression/classification.
        This method must be implemented in child class.

        Returns:
        ----------
        base_learners_: list[str]
            The list containing default base learners' names.
            For regression, will be:
                []
            For classification, will be:
                []
        """
        
        pass

    @abstractmethod
    def _get_default_meta_learner(self) -> str:
        """
        Get the default meta learner for regression/classification.
        This method must be implmented in child class.

        Returns:
        ----------
        meta_learner_: str 
            For regression, will be 'ridge'.
            For classification, will be ''.
        """
        
        pass

    def _validate_and_normalize_base_learners(self):
        """
        Turn base_learners into a usable list of model objects.
        """
        
        self.base_learners_ = {}
        if self.base_learners == 'auto':
            model_list = self._get_default_base_learners()
            for model in model_list:
                self.base_learners_[model] = clone(BASE_LEARNER_REGISTRY[model])
        elif isinstance(self.base_learners, list):
            for model in self.base_learners:
                self.base_learners_[model] = clone(BASE_LEARNER_REGISTRY[model])
        elif isinstance(self.base_learners, dict):
            self.base_learners_ = self.base_learners

    def _validate_and_normalize_meta_learner(self):
        """
        Turn meta_learner into a usable model object.
        """

        self.meta_learner_ = ''
        if self.meta_learner == 'auto':
            model = self._get_default_meta_learner()
            self.meta_learner_ = clone(META_LEARNER_REGISTRY[model])
        else:
            if isinstance(self.meta_learner, str):
                self.meta_learner_ = clone(META_LEARNER_REGISTRY[self.meta_learner])
            else:
                self.meta_learner_ = clone(self.meta_learner)
            

    def _tune_base_learners(self, X, y):
        """
        Only used when tune_base is True.
        Tuning for each base learner.

        Parameters:
        ----------
        X: np.ndarray of shape [n_samples, n_features]
            Predictor matrix.

        y: np.ndarray of shape [n_samples, ] 
            Observed dependent variables.
        """
        
        raise NotImplementedError("Tuning for base learners has not been implemented yet.")

    # key function
    def _generate_oof_matrix(self, X: np.ndarray, y: np.ndarray):
        """
        Generate the OOF matrix for stacking.

        Parameters:
        ----------
        X: np.ndarray of shape [n_samples, n_features]
            Predictor matrix.

        y: np.ndarray of shape [n_samples, ] 
            Observed dependent variables.
        """

        # split the data
        kf = KFold(n_splits = self.kf, shuffle = True, random_state = self.random_state)
        splits = list(kf.split(X, y))

        # initialize out-of-fold prediction matrix
        oof_matrix = np.zeros((y.shape[0], len(self.base_learners_)))

        for col, (name, proto) in enumerate(self.base_learners_.items()):
            for (train_idx, pred_idx) in splits:
                # clone model, so that registered prototypes are not polluted
                learner = clone(proto)

                # split train and predict data
                X_train, y_train, X_pred = X[train_idx, :], y[train_idx], X[pred_idx, :]
                oof_matrix[pred_idx, col] = learner.fit(X_train, y_train).predict(X_pred)

            # train full model and store for final stacking
            full_model = clone(proto)
            full_model.fit(X,y)

            self.full_models_[name] = full_model

        return oof_matrix


    @abstractmethod
    def _prepare_meta_features(self, oof_matrix: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Method specific to stacking type.
        Regular stacking will simply return oof_matrix.
        Super learner will add X to oof_matrix.

        Parameters:
        ----------
        oof_matrix: np.ndarray of shape [n_samples, n_base_learners]
            OOF prediction matrix.
        
        X: np.ndarray of shape [n_samples, n_features]
            Predictor matrix.

        Returns:
        ----------
        meta_features: np.ndarray
            Design matrix for meta learner.
        """
        
        pass

    def _bootstrap_meta_features(self, n_samples: int, bootstrap_iter: int) -> np.ndarray:
        """
        Bootstrapping meta feaeture matrix.

        Parameters:
        ----------
        n_samples: int
            Number of samples.

        bootstrap_iter: int
            Number of times of bootstrapping.

        Returns:
        ----------
        bootstrap_idx: np.ndarray of shape [bootstrap_iter, n_samples]
            bootstrap_idx[i, j] is a row index for the j-th
            sample of the i-th bootstrapping
        """

        return self.rng_.choice(n_samples, size = (bootstrap_iter, n_samples), replace = True)

    @abstractmethod
    def _calculate_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for base learners
        and meta learner and return a df.

        Returns:
        ----------
        basic_table: pd.DataFrame of shape [n_base_learners+1, 4]
            basic_table[i, j] is the j-th metric value
            for the i-th learner (i=1, meta_learner)
        """

        pass

    @abstractmethod
    def _generate_summary_table(self, summary_level: str, bootstrap_idx: np.ndarray) -> pd.DataFrame | dict:
        """
        Store statistical test results
        """

        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the stacker.

        Parameters:
        ----------
        X: np.ndarray of shape [n_sampels, n_features]
            Training data.

        y: np.ndarray of shape [n_samples, ]
            Training dependent variables.
        """

        # get base learners and meta learner ready
        self._validate_and_normalize_base_learners()
        self._validate_and_normalize_meta_learner()
        self.full_models_ = {}

        # generate oof matrix
        self.oof_matrix_ = self._generate_oof_matrix(X, y)

        # prepare meta features
        meta_features = self._prepare_meta_features(self.oof_matrix_, X)

        # store reusable data
        self.meta_features_ = meta_features
        self.y_ = y

        # bootstrapping and store as cache
        self.rng_ = np.random.default_rng(seed = self.random_state)
        self.bootstrap_idx_cache_ = self._bootstrap_meta_features(y.shape[0], self.bootstrap_iter)

        # fit meta learner
        self.meta_learner_fitted_ = clone(self.meta_learner_).fit(meta_features, y)

        # calculate basic metrics
        self.basic_metrics_ = self._calculate_metrics()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using fitted meta learner.

        Parameters:
        ----------
        X: np.ndarray of shape [n_samples, n_features]
            New data used for prediction.

        Returns:
        ----------
        pred: np.ndarray of shape [n_samples,]
            Predicted values using meta learner.
        """

        # initialize prediction matrix containing each base learners' predictions
        pred_base = np.zeros((X.shape[0], len(self.full_models_)))

        # base learner prediction and generate pred_base
        for idx, model in enumerate(self.full_models_.values()):
            preds = model.predict(X)
            pred_base[:, idx] = preds

        # generate pred_meta for meta learner prediction
        pred_meta = self._prepare_meta_features(pred_base, X)

        # meta learner prediction
        return self.meta_learner_fitted_.predict(pred_meta)

    @abstractmethod
    def summary(self, summary_level: str = 'diagnostic', bootstrap_iter: None | int = None, return_df: bool = False) -> None | pd.DataFrame:
        """
        Different class will generate
        difference summary based on
        different summary_level.

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

        return_df: bool
            Whether to save the summary as a dataframe.
            Default to be False.

        Returns:
        ----------
        summary_df: pd.DataFrame
            Table containing the summary.
        """

        pass
        
            


    
            

        
    