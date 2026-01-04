# NBA Game Prediction Project

This project uses machine learning (XGBoost) to predict the winners of NBA games. It leverages historical data, advanced statistical features, and rolling averages to forecast game outcomes with a focus on the 2024-25 season.

## Project Overview

The goal of this project is to build a robust model that can predict whether the **Home Team** will win a given matchup. The model is trained on data from the 2021-22, 2022-23, and 2023-24 seasons and tested on the current 2024-25 season.

## Key Features

*   **Data Collection**: Automated data fetching using the `nba_api` to get game logs, player stats, and team standings.
*   **Advanced Metrics**: Calculates advanced basketball stats such as:
    *   **Possessions (POSS)**
    *   **True Shooting Percentage (TS%)**
    *   **Effective Field Goal Percentage (eFG%)**
    *   **Net Rating (NET_RAT)**
*   **Feature Engineering**:
    *   **Rolling Averages**: Uses 10-game exponential moving averages for key stats to capture current team form.
    *   **Fatigue & Rest**: Accounts for days of rest, back-to-back games, and schedule density (3 games in 4 days, 4 games in 5 days).
    *   **Star Player Availability**: Tracks whether a team's top scorer is active for the game.
    *   **Strength of Schedule**: Adjusts expectations based on the quality of opponents faced recently.
*   **Modeling**:
    *   Uses **XGBoost Classifier** for prediction.
    *   Hyperparameter tuning via `GridSearchCV`.
    *   Achieves ~67% accuracy on the 2024-25 season (as of latest test).
    *   **High Confidence Predictions**: Accuracy increases to >76% when the model's confidence is above 70%.

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_cleaning.ipynb    # Data fetching, feature engineering, and model training
│   ├── 02_2425_season_test.ipynb # Testing the model on the 2024-25 season
│   ├── nba_model_2024.pkl        # Trained XGBoost model
│   ├── model_features.pkl        # List of features used by the model
│   └── iso_model.pkl             # (Optional) Isolation Forest model
├── test_diff.py                  # Simple test script for pandas functionality
└── requirements.txt              # Project dependencies
```

## Installation

1.  Clone the repository.
2.  Install the required Python packages. You can use the following command:

```bash
pip install pandas numpy nba_api xgboost scikit-learn joblib plotly
```

## Usage

1.  **Training**: Run `notebooks/01_data_cleaning.ipynb` to fetch historical data, process features, and train the model. This will save the trained model to `nba_model_2024.pkl`.
2.  **Prediction**: Run `notebooks/02_2425_season_test.ipynb` to load the trained model and generate predictions for the current 2024-25 season.

## Model Performance

*   **Overall Accuracy**: ~67%
*   **High Confidence Accuracy (>70%)**: ~76%

## Future Improvements

*   Integrate more granular player-level data.
*   Experiment with different rolling windows (e.g., 5 games vs 15 games).
*   Add more "hustle" stats (deflections, loose balls recovered).
*   Deploy as a web app or daily prediction script.
