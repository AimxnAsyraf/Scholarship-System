import os
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, classification_report

# ---------------------------------------------------------
# 1. Fuzzy Logic Helper Functions & Transformer
# ---------------------------------------------------------

def triangular_mf(x, a, b, c):
    """
    Triangular membership function.
    a: start, b: peak, c: end
    """
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-6), (c - x) / (c - b + 1e-6)))

class FuzzyTransformer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to add Fuzzy Logic features for CGPA, Income, and Score.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure input is DataFrame for column access, though Pipeline might pass arrays
        # If array, we assume column order: CGPA (0), Income (1), Score (2), etc.
        # Ideally, we use this on specific columns via ColumnTransformer.
        
        # If X is a numpy array (from previous steps in pipeline), convert to DataFrame for readability
        # Note: In this script, we will apply FuzzyTransformer to specific numerical columns
        
        # CGPA Fuzzy Sets (Range approx 0-4.0)
        # Low: 0-2.5, Medium: 2.0-3.5, High: 3.0-4.0
        if isinstance(X, np.ndarray):
             # Assuming order: CGPA, Income, Score
            cgpa = X[:, 0]
            income = X[:, 1]
            score = X[:, 2]
        else:
            cgpa = X['CGPA'].values
            income = X['Family Income per Month (RM)'].values
            score = X['Co-curricular Score (/100)'].values

        # CGPA Features
        cgpa_low = triangular_mf(cgpa, 0.0, 0.0, 2.5)
        cgpa_med = triangular_mf(cgpa, 2.0, 3.0, 3.8)
        cgpa_high = triangular_mf(cgpa, 3.2, 4.0, 4.0)

        # Income Features (Range approx 0 - 20000)
        # Low: <5k, Med: 3k-12k, High: >10k
        inc_low = triangular_mf(income, 0, 0, 5000)
        inc_med = triangular_mf(income, 3000, 7500, 12000)
        inc_high = triangular_mf(income, 8000, 20000, 20000)

        # Score Features (Range 0-100)
        score_low = triangular_mf(score, 0, 0, 50)
        score_med = triangular_mf(score, 30, 60, 90)
        score_high = triangular_mf(score, 70, 100, 100)

        # Stack new features
        fuzzy_features = np.column_stack([
            cgpa_low, cgpa_med, cgpa_high,
            inc_low, inc_med, inc_high,
            score_low, score_med, score_high
        ])
        
        # Concatenate original X with fuzzy features
        if isinstance(X, pd.DataFrame):
            return np.hstack([X.values, fuzzy_features])
        else:
            return np.hstack([X, fuzzy_features])

# ---------------------------------------------------------
# 2. Load and Prepare Data
# ---------------------------------------------------------

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    
    # Target Mapping (Ordinal Encoding for Regression)
    target_map = {'Ineligible': 0, 'In Review': 1, 'Eligible': 2}
    df['Target_Score'] = df['Eligibility'].map(target_map)
    
    # Drop original target column for X
    X = df.drop(columns=['Eligibility', 'Target_Score'])
    y = df['Target_Score']
    
    return X, y

# ---------------------------------------------------------
# 3. Main Training Script
# ---------------------------------------------------------

def train_pipeline():
    print("Loading data...")
    X, y = load_and_prep_data('scholarship_dataset.csv')

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Columns
    numeric_features = ['CGPA', 'Family Income per Month (RM)', 'Co-curricular Score (/100)']
    # We include Leadership in numeric or pass through
    other_numeric = ['Number of Leadership Positions'] 
    categorical_features = ['Field of Study']

    # Preprocessing Pipeline
    # 1. Encode Categorical
    # 2. Apply Fuzzy Logic to specific Numeric cols
    # 3. Scale everything
    
    # Step 1: Handle Categorical
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Step 2: Handle Numeric (Fuzzy + Scale)
    # We need a custom pipeline for numeric: first generate fuzzy, then scale
    numeric_transformer = Pipeline(steps=[
        ('fuzzy', FuzzyTransformer()), # Adds fuzzy cols to the numeric inputs
        ('scaler', StandardScaler())
    ])
    
    # Combine transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('other', StandardScaler(), other_numeric), # Just scale leadership
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create Full Pipeline with Gradient Boosting
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])

    # Create Full Pipeline with ElasticNet (Baseline)
    en_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(random_state=42))
    ])

    # ---------------------------------------------------------
    # 4. Training & Evaluation
    # ---------------------------------------------------------
    
    print("\n--- Training Gradient Boosting Regressor ---")
    gb_pipeline.fit(X_train, y_train)
    gb_preds = gb_pipeline.predict(X_test)
    
    print(f"GB MAE: {mean_absolute_error(y_test, gb_preds):.4f}")
    print(f"GB RMSE: {np.sqrt(mean_squared_error(y_test, gb_preds)):.4f}")
    
    # Convert Regression output to Classification for F1
    gb_preds_class = np.round(gb_preds).clip(0, 2) # Round to nearest int class
    print(f"GB F1 Score (weighted): {f1_score(y_test, gb_preds_class, average='weighted'):.4f}")

    print("\n--- Training ElasticNet Baseline ---")
    en_pipeline.fit(X_train, y_train)
    en_preds = en_pipeline.predict(X_test)
    
    print(f"ElasticNet MAE: {mean_absolute_error(y_test, en_preds):.4f}")
    print(f"ElasticNet RMSE: {np.sqrt(mean_squared_error(y_test, en_preds)):.4f}")
    
    en_preds_class = np.round(en_preds).clip(0, 2)
    print(f"ElasticNet F1 Score (weighted): {f1_score(y_test, en_preds_class, average='weighted'):.4f}")

    # ---------------------------------------------------------
    # 5. Saving the Model
    # ---------------------------------------------------------
    
    # Create directory if not exists
    save_dir = 'fastapi_service/model'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'scholarship_pipeline.joblib')
    
    print(f"\nSaving Gradient Boosting pipeline to {save_path}...")
    joblib.dump(gb_pipeline, save_path)
    print("Done.")

if __name__ == "__main__":
    train_pipeline()