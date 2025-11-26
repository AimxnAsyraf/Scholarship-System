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

# Fuzzy Logic Transformer for feature engineering
class FuzzyLogicTransformer(BaseEstimator, TransformerMixin):
    """
    Apply fuzzy logic rules to create fuzzy membership scores
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # CGPA Fuzzy Membership (0-4.0 scale)
        X['cgpa_low'] = np.where(X['CGPA'] <= 2.5, 1.0, 
                                 np.where(X['CGPA'] <= 3.0, (3.0 - X['CGPA']) / 0.5, 0.0))
        X['cgpa_medium'] = np.where((X['CGPA'] > 2.5) & (X['CGPA'] <= 3.5),
                                    np.minimum((X['CGPA'] - 2.5) / 0.5, (3.5 - X['CGPA']) / 0.5), 0.0)
        X['cgpa_high'] = np.where(X['CGPA'] >= 3.5, 1.0,
                                  np.where(X['CGPA'] >= 3.0, (X['CGPA'] - 3.0) / 0.5, 0.0))
        
        # Family Income Fuzzy Membership (normalize to 0-15000)
        income = X['Family Income per Month (RM)']
        X['income_low'] = np.where(income <= 5000, 1.0,
                                   np.where(income <= 8000, (8000 - income) / 3000, 0.0))
        X['income_medium'] = np.where((income > 5000) & (income <= 10000),
                                      np.minimum((income - 5000) / 3000, (10000 - income) / 2000), 0.0)
        X['income_high'] = np.where(income >= 10000, 1.0,
                                    np.where(income >= 8000, (income - 8000) / 2000, 0.0))
        
        # Co-curricular Score Fuzzy Membership (0-100)
        cocurr = X['Co-curricular Score (/100)']
        X['cocurr_low'] = np.where(cocurr <= 40, 1.0,
                                   np.where(cocurr <= 60, (60 - cocurr) / 20, 0.0))
        X['cocurr_medium'] = np.where((cocurr > 40) & (cocurr <= 80),
                                      np.minimum((cocurr - 40) / 20, (80 - cocurr) / 20), 0.0)
        X['cocurr_high'] = np.where(cocurr >= 80, 1.0,
                                    np.where(cocurr >= 60, (cocurr - 60) / 20, 0.0))
        
        # Leadership Fuzzy Membership
        leadership = X['Number of Leadership Positions']
        X['leadership_low'] = np.where(leadership == 0, 1.0, 0.0)
        X['leadership_medium'] = np.where((leadership >= 1) & (leadership <= 2), 1.0, 0.0)
        X['leadership_high'] = np.where(leadership >= 3, 1.0, 0.0)
        
        # Composite Fuzzy Score
        X['fuzzy_eligibility_score'] = (
            0.4 * X['cgpa_high'] + 
            0.2 * X['income_low'] + 
            0.25 * X['cocurr_high'] + 
            0.15 * X['leadership_high']
        )
        
        return X

def evaluate_model(pipeline, X_test, y_test, model_name):
    print(f"\n{'='*60}")
    print(f"{model_name} Evaluation")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_rounded = np.round(y_pred).clip(0, 2)  # Round and clip to valid range
    
    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Classification metrics (treating as multiclass)
    print(f"\nClassification Metrics (rounded predictions):")
    print(f"  Accuracy: {np.mean(y_pred_rounded == y_test):.4f}")
    print(f"  F1-Score (macro): {f1_score(y_test, y_pred_rounded, average='macro'):.4f}")
    print(f"  F1-Score (weighted): {f1_score(y_test, y_pred_rounded, average='weighted'):.4f}")
    
    print(f"\nClassification Report:")
    target_names = ['Ineligible', 'In Review', 'Eligible']
    print(classification_report(y_test, y_pred_rounded, target_names=target_names))
    
    return mae, rmse, y_pred

def main():
    print("="*60)
    print("Scholarship Eligibility Model Training")
    print("="*60)
    
    # Load the dataset
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'scholarship_dataset.csv')
    df = pd.read_csv(dataset_path)
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Eligibility'].value_counts()}")
    
    # Prepare features and target
    # Map eligibility to numeric: Eligible=2, In Review=1, Ineligible=0
    eligibility_map = {'Eligible': 2, 'In Review': 1, 'Ineligible': 0}
    df['Eligibility_Numeric'] = df['Eligibility'].map(eligibility_map)
    
    # Features and target
    X = df.drop(['Eligibility', 'Eligibility_Numeric'], axis=1)
    y = df['Eligibility_Numeric']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Define categorical and numerical columns
    categorical_features = ['Field of Study']
    numerical_features = ['CGPA', 'Family Income per Month (RM)', 
                         'Co-curricular Score (/100)', 'Number of Leadership Positions']
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Train Gradient Boosting Regressor
    print("\n" + "="*60)
    print("Training Gradient Boosting Regressor...")
    print("="*60)
    
    gb_pipeline = Pipeline(steps=[
        ('fuzzy', FuzzyLogicTransformer()),
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=1
        ))
    ])
    
    gb_pipeline.fit(X_train, y_train)
    
    # Train ElasticNet baseline
    print("\n" + "="*60)
    print("Training ElasticNet baseline...")
    print("="*60)
    
    elasticnet_pipeline = Pipeline(steps=[
        ('fuzzy', FuzzyLogicTransformer()),
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet(
            alpha=0.5,
            l1_ratio=0.5,
            random_state=42,
            max_iter=5000
        ))
    ])
    
    elasticnet_pipeline.fit(X_train, y_train)
    
    # Evaluate both models
    gb_mae, gb_rmse, _ = evaluate_model(gb_pipeline, X_test, y_test, "Gradient Boosting")
    en_mae, en_rmse, _ = evaluate_model(elasticnet_pipeline, X_test, y_test, "ElasticNet")
    
    # Model comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(f"\n{'Model':<25} {'MAE':<12} {'RMSE':<12}")
    print("-" * 60)
    print(f"{'Gradient Boosting':<25} {gb_mae:<12.4f} {gb_rmse:<12.4f}")
    print(f"{'ElasticNet':<25} {en_mae:<12.4f} {en_rmse:<12.4f}")
    
    # Select best model
    best_model = gb_pipeline if gb_mae < en_mae else elasticnet_pipeline
    best_model_name = "Gradient Boosting" if gb_mae < en_mae else "ElasticNet"
    print(f"\n✅ Best Model: {best_model_name}")
    
    # Save the best model
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'backend', 'model')
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'scholarship_pipeline.joblib')
    joblib.dump(best_model, model_path)
    
    print(f"\n{'='*60}")
    print(f"Model saved successfully!")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Model type: {best_model_name}")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.2f} KB")
    
    # Save feature names for reference
    feature_info = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'target_mapping': {'Ineligible': 0, 'In Review': 1, 'Eligible': 2},
        'model_name': best_model_name,
        'mae': float(gb_mae if best_model_name == "Gradient Boosting" else en_mae),
        'rmse': float(gb_rmse if best_model_name == "Gradient Boosting" else en_rmse)
    }
    
    feature_info_path = os.path.join(output_dir, 'model_info.joblib')
    joblib.dump(feature_info, feature_info_path)
    print(f"Model info saved to: {feature_info_path}")
    
    # Test the saved model
    print("\n" + "="*60)
    print("Testing Saved Model")
    print("="*60)
    
    loaded_model = joblib.load(model_path)
    
    # Test prediction on a sample
    sample = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample)
    prediction_class = int(np.round(prediction[0]).clip(0, 2))
    
    eligibility_labels = {0: 'Ineligible', 1: 'In Review', 2: 'Eligible'}
    print(f"\nSample input:")
    for col, val in sample.iloc[0].items():
        print(f"  {col}: {val}")
    print(f"\nPrediction: {prediction[0]:.4f}")
    print(f"Predicted class: {eligibility_labels[prediction_class]}")
    print(f"Actual class: {eligibility_labels[y_test.iloc[0]]}")
    
    print("\n✅ Model training and saving complete!")

if __name__ == "__main__":
    main()
