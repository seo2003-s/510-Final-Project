
#Data sources
KAGGLE_DATASET_FRAMINGHAM = "aasheesh200/framingham-heart-study-dataset"
KAGGLE_DATASET_HEART = "fedesoriano/heart-failure-prediction"

FRAMINGHAM_CACHE_PATH = (
    "~/.cache/kagglehub/datasets/aasheesh200/"
    "framingham-heart-study-dataset/versions/1/framingham.csv"
)
HEART_CACHE_PATH = (
    "~/.cache/kagglehub/datasets/fedesoriano/"
    "heart-failure-prediction/versions/1/heart.csv"
)

CARDIO_CSV_FILENAME = "Cardiovascular_Disease_Dataset.csv"

#Column name mappings
CARDIO_RENAME = {
    "gender": "sex",
    "serumcholestrol": "cholesterol",
    "restingBP": "resting_bp",
}

FRAMINGHAM_RENAME = {
    "male": "sex",
    "totChol": "cholesterol",
    "sysBP": "resting_bp",
    "TenYearCHD": "target",
}

HEART_RENAME = {
    "Age": "age",
    "Sex": "sex",
    "Cholesterol": "cholesterol",
    "RestingBP": "resting_bp",
    "HeartDisease": "target",
}

HEART_SEX_MAP = {"M": 1, "F": 0}

#Common feature columns
COMMON_COLS = ["age", "sex", "cholesterol", "resting_bp", "target"]
FEATURE_COLS = ["age", "sex", "cholesterol", "resting_bp"]
NUMERIC_COLS = ["age", "cholesterol", "resting_bp"]
TARGET_COL = "target"
SOURCE_COL = "source"

SOURCE_CARDIO = "cardio"
SOURCE_FRAMINGHAM = "framingham"
SOURCE_HEART = "heart"

#Modelling variables
TEST_SIZE = 0.2
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.3
MAX_ITER = 1000

# Output paths
PLOT_LINEARITY = "linearity_plots.png"