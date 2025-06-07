"""
Hyperparameter grids for different regression models.
Used with sklearn.model_selection.ParameterGrid
"""

from typing import Dict, List

# Ridge regression grid
ridge_param_grid: Dict[str, List] = {
    'alpha': [0.1, 1.0, 10.0],
    'fit_intercept': [True, False],
}

# ElasticNet regression grid
elasticnet_param_grid: Dict[str, List] = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.8],
    'fit_intercept': [True, False],
}

# XGBoost regression grid
# xgb_param_grid: Dict[str, List] = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 4, 5],
#     'min_child_weight': [1, 2, 3],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2],
#     'reg_alpha': [0, 0.1, 1.0],
#     'reg_lambda': [0, 0.1, 1.0],
# }

xgb_param_grid: Dict[str, List] = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01],
    'max_depth': [3],
    'min_child_weight': [1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0],
    'reg_alpha': [0, 1.0],
    'reg_lambda': [1.0],
}

# Master dictionary for access by name
param_grids: Dict[str, Dict[str, List]] = {
    "ridge": ridge_param_grid,
    "elasticnet": elasticnet_param_grid,
    "xgboost": xgb_param_grid,
}
