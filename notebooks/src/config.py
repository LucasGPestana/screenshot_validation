import os

PROJECT_DIRPATH = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIRPATH = os.path.join(
    PROJECT_DIRPATH,
    "data"
)

ORIGINAL_DATA_FILEPATH = os.path.join(
    DATA_DIRPATH,
    "features_completas.csv"
)

CLEANED_DATA_FILEPATH = os.path.join(
    DATA_DIRPATH,
    "features_completas_cleaned.parquet"
)

MODELS_DIRPATH = os.path.join(
    PROJECT_DIRPATH,
    "models"
)

FINAL_MODEL_FILEPATH = os.path.join(
    MODELS_DIRPATH,
    "decision_tree_preprocessing.joblib"
)