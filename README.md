# Energy Prices Prediction

This repository contains code for predicting real-time energy prices using deep learning models, specifically LSTM and Transformer architectures. 

## Key Findings

- EDA shows seasonality in daily, weekly and yearly trends.
- The both models' performance is comparable.
- Transformer takes longer to train for the same results.
- Further testing can potentially improve the models' performance.

## Usage

1. Install the required dependencies: `pip install -r requirements.txt`
2. Set TRAIN = False to use the already trained models and their performance metrics.
3. Set TRAIN = True to train the models and evaluate their performance.

## Files

1. `data.py` contains the code for preprocessing the data.
2. `models.py` contains the code for the LSTM and Transformer models.
3. `main.ipynb` is the main notebook for EDA, training and evaluation.
4. `saved_models` contains the recently trained models. 
4. `saved_models_backup` contains the backup/best trained models. 

## Note

1. The training time for the models is long(~40 mins for both on a 6GB GPU), especially for the Transformer model.
2. Use saved_models_backup if limited training times don't allow for complete training.