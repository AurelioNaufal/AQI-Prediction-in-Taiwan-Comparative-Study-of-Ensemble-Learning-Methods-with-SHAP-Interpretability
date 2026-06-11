import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'air_quality_2024.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

for _d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Optuna
OPTUNA_TRIALS    = 30
OPTUNA_TIMEOUT   = 360   # seconds per model
OPTUNA_SUBSAMPLE = 0.25  # fraction of train rows used during each Optuna trial

# LSTM / NN
LOOKBACK        = 24   # hours lookback window
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 256
NN_EPOCHS       = 60
NN_BATCH_SIZE   = 512

NUMERIC_FEATURES = [
    'so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5',
    'no2', 'nox', 'no', 'windspeed', 'winddirec',
    'co_8hr', 'pm2.5_avg', 'pm10_avg', 'so2_avg',
]
TARGET      = 'aqi'
CAT_FEATURE = 'pollutant'

# Clean category labels used after mapping
POLLUTANT_MAP = {
    'PM2.5':                   'PM25',
    'PM10':                    'PM10',
    'Ozone (8hr)':             'Ozone_8hr',
    'Ozone':                   'Ozone',
    'Nitrogen Dioxide (NO2)':  'NO2',
    'Sulfur Dioxide (SO2)':    'SO2',
}
POLLUTANT_NONE_LABEL = 'None'
