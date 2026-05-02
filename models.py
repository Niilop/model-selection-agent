from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

CANDIDATES = {
    "linear": {
        "cls": LinearRegression,
        "fixed_kwargs": {},
        "grid": {},
        "description": "Ordinary least squares — fast, interpretable, assumes linear relationships",
    },
    "ridge": {
        "cls": Ridge,
        "fixed_kwargs": {},
        "grid": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        "description": "L2-regularised linear model — handles multicollinearity well",
    },
    "lasso": {
        "cls": Lasso,
        "fixed_kwargs": {"max_iter": 10_000},
        "grid": {"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
        "description": "L1-regularised linear model — performs implicit feature selection",
    },
    "decision_tree": {
        "cls": DecisionTreeRegressor,
        "fixed_kwargs": {"random_state": 42},
        "grid": {
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [1, 3, 10],
        },
        "description": "Single CART tree — captures non-linearities, prone to overfit",
    },
    "random_forest": {
        "cls": RandomForestRegressor,
        "fixed_kwargs": {"random_state": 42},
        "grid": {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_leaf": [1, 2, 5],
        },
        "description": "Bagged ensemble of trees — robust, low variance, slower to train",
    },
    "gradient_boosting": {
        "cls": GradientBoostingRegressor,
        "fixed_kwargs": {"random_state": 42},
        "grid": {
            "n_estimators": [100, 300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.7, 0.8, 1.0],
        },
        "description": "Sequential boosting ensemble — often best accuracy, highest training cost",
    },
}
