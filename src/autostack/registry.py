import warnings
from sklearn.linear_model import (
    LinearRegression, 
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNet,
    ElasticNetCV,
    BayesianRidge
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor

# Base LEARNER Registry
BASE_LEARNER_REGISTRY = {}

# Linear models
BASE_LEARNER_REGISTRY['lr'] = LinearRegression()
BASE_LEARNER_REGISTRY['ridge'] = Ridge(alpha = 1, random_state = 42)
BASE_LEARNER_REGISTRY['lasso'] = Lasso(alpha = 1, random_state = 42)
BASE_LEARNER_REGISTRY['enet'] = ElasticNet(alpha = 1, l1_ratio = 0.5, random_state = 42)

# NearestNeighbors and Supper Vector Machine
BASE_LEARNER_REGISTRY['knn'] = KNeighborsRegressor(n_neighbors = 5)
BASE_LEARNER_REGISTRY['svr'] = SVR(kernel = 'rbf', C = 1, gamma = 'scale')

# Tree model
BASE_LEARNER_REGISTRY['dt'] = DecisionTreeRegressor(max_depth = None, random_state = 42)

# Bagging models
BASE_LEARNER_REGISTRY['rf'] = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state = 42, n_jobs = -1)
BASE_LEARNER_REGISTRY['et'] = ExtraTreesRegressor(n_estimators = 100, max_depth = None, random_state = 42, n_jobs = -1)

# Boosting models
BASE_LEARNER_REGISTRY['gbm'] = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)
BASE_LEARNER_REGISTRY['hist_gbm'] = HistGradientBoostingRegressor(max_iter = 100, random_state = 42)
BASE_LEARNER_REGISTRY['ada'] = AdaBoostRegressor(n_estimators = 100, random_state = 42)

# Neural network
BASE_LEARNER_REGISTRY['mlp'] = MLPRegressor(hidden_layer_sizes = (64, 32), max_iter = 500, random_state = 42)

# other models require extra installation
try:
    from xgboost import XGBRegressor
    BASE_LEARNER_REGISTRY['xgb'] = XGBRegressor(n_estimators = 100, max_depth = 3, random_state = 42, n_jobs = -1)
except ImportError:
    warnings.warn("XGBoost not installed. Skipping 'xgb' in BASE_LEARNER_REGISTRY.")

try:
    from lightgbm import LGBMRegressor
    BASE_LEARNER_REGISTRY['lgbm'] = LGBMRegressor(n_estimators = 100, max_depth = -1, random_state = 42, n_jobs = -1)
except ImportError:
    warnings.warn("LightGBM not installed. Skipping 'lgbm' in BASE_LEARNER_REGISTRY.")

try:
    from catboost import CatBoostRegressor
    BASE_LEARNER_REGISTRY['cat'] = CatBoostRegressor(n_estimators = 100, depth = 3, random_state = 42, verbose = 0)
except ImportError:
    warnings.warn("CatBoost not installed. Skipping 'cat' in BASE_LEARNER_REGISTRY.")

# Meta Regressor Registry
META_LEARNER_REGISTRY = {}

# Linear models
META_LEARNER_REGISTRY['lr'] = LinearRegression()

# Penalized linear models
META_LEARNER_REGISTRY['ridge'] = Ridge(alpha = 1, random_state = 42)
META_LEARNER_REGISTRY['ridge_cv'] = RidgeCV(alphas = (0.01, 0.1, 1, 10))
META_LEARNER_REGISTRY['lasso'] = Lasso(alpha = 1, random_state = 42)
META_LEARNER_REGISTRY['lasso_cv'] = LassoCV(cv = 5, random_state = 42)
META_LEARNER_REGISTRY['enet'] = ElasticNet(alpha = 1, l1_ratio = 0.5, random_state = 42)
META_LEARNER_REGISTRY['enet_cv'] = ElasticNetCV(cv = 5, random_state = 42)

# Non-negative linear models
META_LEARNER_REGISTRY['nnls'] = LinearRegression(positive = True)
META_LEARNER_REGISTRY['nn_ridge'] = Ridge(alpha = 1, random_state = 42, positive = True)
META_LEARNER_REGISTRY['nn_lasso'] = Lasso(alpha = 1, random_state = 42, positive = True)
META_LEARNER_REGISTRY['nn_enet'] = ElasticNet(alpha = 1, l1_ratio = 0.5, random_state = 42, positive = True)

# Tree model
#META_LEARNER_REGISTRY['dt'] = DecisionTreeRegressor(max_depth = None, random_state = 42)

"""
# Bagging models
META_LEARNER_REGISTRY['rf'] = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state = 42, n_jobs = -1)
META_LEARNER_REGISTRY['et'] = ExtraTreesRegressor(n_estimators = 100, max_depth = None, random_state = 42, n_jobs = -1)

# Boosting models
META_LEARNER_REGISTRY['gbm'] = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)
META_LEARNER_REGISTRY['hist_gbm'] = HistGradientBoostingRegressor(max_iter = 100, random_state = 42)
META_LEARNER_REGISTRY['ada'] = AdaBoostRegressor(n_estimators = 100, random_state = 42)

# Support Vector Machine and neural network
META_LEARNER_REGISTRY['svr'] = SVR(kernel = 'rbf', C = 1, gamma = 'scale')
META_LEARNER_REGISTRY['mlp'] = MLPRegressor(hidden_layer_sizes = (64, 32), max_iter = 500, random_state = 42)

# other models require extra installation
try:
    from xgboost import XGBRegressor
    META_LEARNER_REGISTRY['xgb'] = XGBRegressor(n_estimators = 100, max_depth = 3, random_state = 42, n_jobs = -1)
except ImportError:
    warnings.warn("XGBoost not installed. Skipping 'xgb' in META_MODEL_REGISTRY.")

try:
    from lightgbm import LGBMRegressor
    META_LEARNER_REGISTRY['lgbm'] = LGBMRegressor(n_estimators = 100, max_depth = -1, random_state = 42, n_jobs = -1)
except ImportError:
    warnings.warn("LightGBM not installed. Skipping 'lgbm' in META_MODEL_REGISTRY.")

try:
    from catboost import CatBoostRegressor
    META_LEARNER_REGISTRY['cat'] = CatBoostRegressor(n_estimators = 100, depth = 3, random_state = 42, verbose = 0)
except ImportError:
    warnings.warn("CatBoost not installed. Skipping 'cat' in META_MODEL_REGISTRY.")

try:
    from pygam import LinearGAM
    META_LEARNER_REGISTRY['gam'] = LinearGAM()
except ImportError:
    warnings.warn("pygam not installed. Skipping 'gam' in META_LEARNER_REGISTRY")
"""