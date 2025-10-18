import pandas as pd


def count_outliers(df : pd.DataFrame, column: str) -> int:

    """Conta a quantidade de outliers presente na coluna

    Parameters
    ----------
    df : pandas.DataFrame
        Base que contém a coluna
    column : str
        Nome da coluna que se deseja contar os outliers
    
    Returns
    -------
    int
        Quantidade de outliers superiores e inferiores na coluna
    
    """

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    upper_thres = q3 + 1.5 * iqr
    lower_thres = q1 - 1.5 * iqr

    return df[
        (df[column] < lower_thres) | (df[column] > upper_thres)
    ].shape[0]


def organize_results(results):

    for model in results.keys():

        results[model]["time_seconds"] = results[model]["fit_time"] + results[model]["score_time"]

    df_results = pd.DataFrame(results).T

    df_results = df_results.explode(df_results.columns.tolist()).reset_index(names=["model"])

    return df_results