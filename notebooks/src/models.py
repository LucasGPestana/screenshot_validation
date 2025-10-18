import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
)


def construct_pipeline(classificator, preprocessing, pca=None):

    steps = []

    if preprocessing:

        steps.append(("preprocessing", preprocessing))
    
    if pca:

        steps.append(("pca", pca))
    
    steps.append(("clf", classificator))

    pipeline = Pipeline(steps)
    
    return pipeline


def train_validate_classificator(X, y, kf, classificator, preprocessing, pca=None):

    estimator = construct_pipeline(classificator, preprocessing, pca)

    results = cross_validate(
        estimator,
        X,
        y,
        cv=kf,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
    )

    return results

def grid_search_classificator(classificator, preprocessing, param_grid, kf, refit_metric="f1", pca=None):

    estimator = construct_pipeline(classificator, preprocessing, pca)

    grid_search = GridSearchCV(
        estimator,
        param_grid=param_grid,
        cv=kf,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "average_precision",
        ],
        refit=refit_metric,
        n_jobs=1,
        verbose=1,
    )

    return grid_search