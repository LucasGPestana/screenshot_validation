import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
)


import joblib


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

def predict(model_filepath : str, data_filepath : str) -> str:

    """Rotula uma imagem, a partir de seus descritores, em manipulado ou autêntico

    Parameters
    ----------
    model_filepath : str
        Caminho do arquivo do modelo
    
    data_filepath : str
        Caminho com os dados dos descritores da imagem. Deve ter extensão 'csv'.
    
    Returns
    -------
    str
        Rótulo da imagem em 'Manipulado' ou 'Autêntico'
    """

    model = joblib.load(model_filepath)
    data = pd.read_csv(data_filepath)

    # Features categóricas que o modelo foi treinado
    categoric_columns = [
        'FOS_FOS_Median',
        'FOS_FOS_Mode',
        'FOS_FOS_MinimalGrayLevel',
        'FOS_FOS_10Percentile',
        'FOS_FOS_25Percentile',
        'FOS_FOS_75Percentile',
        'FOS_FOS_90Percentile',
        'FOS_FOS_HistogramWidth',
        'cor_R_mediana',
        'cor_G_min',
        'cor_G_mediana',
        'cor_B_min',
        'cor_B_mediana',
    ]

    # Features numéricas que o modelo foi treinado
    numeric_columns = [
        'FOS_FOS_Variance',
        'GLCM_GLCM_ASM_Mean',
        'tex_gradiente_std',
        'tex_gradiente_max',
        'tex_laplacian_media',
        'tex_suavidade',
        'comp_variancia_blocos',
        'comp_media_blocos'
        ]

    # Remove colunas desnecessárias
    data = data.drop(["nome_arquivo", "rotulo"], axis=1)

    # Agrega as colunas categóricas em apenas duas categorias: zero e non_zero
    for column in categoric_columns:

        data[column] = data[column].apply(lambda x: "zero" if x == 0 else "non_zero")

    # Seleciona apenas as features que foram treinadas
    data = data.filter(numeric_columns + categoric_columns)

    return "Manipulado" if model.predict(data)[0] == 1 else "Autêntico"