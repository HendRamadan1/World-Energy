# File paths
RAW_DATA_PATH = r"../data/raw/owid-energy-data.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_energy.csv"
MODEL_SAVE_PATH = "models/saved_models/model.pkl"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 5

# Features and target
FEATURES = [
    'year',
    'gdp',
    'population',
    'renewables_energy_per_capita',
    'fossil_energy_per_capita'
]
TARGET = 'primary_energy_consumption'