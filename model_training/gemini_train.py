#First version
import pandas as pd
import numpy as np
import json
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, f1_score, 
                             precision_score, recall_score, accuracy_score)
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Fuzzy Logic Helper Functions
# ==========================================
def triangular_membership(x, a, b, c):
    """Calculates triangular membership degree."""
    term1 = (x - a) / (b - a + 1e-6)
    term2 = (c - x) / (c - b + 1e-6)
    return np.maximum(0, np.minimum(term1, term2))

def trapezoidal_membership(x, a, b, c, d):
    """Calculates trapezoidal membership degree."""
    term1 = (x - a) / (b - a + 1e-6)
    term2 = (d - x) / (d - c + 1e-6)
    return np.maximum(0, np.minimum(np.minimum(term1, 1), term2))

def apply_fuzzy_logic(df):
    """Applies fuzzy logic transformation."""
    data = df.copy()
    
    # --- Income ---
    data['Fuzzy_Income_Low'] = triangular_membership(data['Household Income'], -1, 0, 12000)  #Min: 12000
    data['Fuzzy_Income_Med'] = triangular_membership(data['Household Income'], 6000, 12000, 70000) #Med: 70000
    data['Fuzzy_Income_High'] = trapezoidal_membership(data['Household Income'], 12000, 27000, 280000, 290000) #Max: 290000
    
    # --- SPM (Number of As) ---
    data['Fuzzy_SPM_Low'] = triangular_membership(data['SPM Result (As)'], -1, 0, 6) #Min: 4
    data['Fuzzy_SPM_Med'] = triangular_membership(data['SPM Result (As)'], 3, 7, 10) #Med:6
    data['Fuzzy_SPM_High'] = trapezoidal_membership(data['SPM Result (As)'], 7, 10, 12, 15) #Max: 15
    
    # --- Unified CGPA Foundation, Undergraduate, STPM, Matriculation ---
    # This avoids the "Zero Problem" where missing STPM is seen as "Low Score"
    col = 'CGPA_Unified'
    data['Fuzzy_CGPA_Low'] = trapezoidal_membership(data[col], -1, 0.0, 2.5, 3.0)
    data['Fuzzy_CGPA_Med'] = triangular_membership(data[col], 2.5, 3.0, 3.5)
    data['Fuzzy_CGPA_High'] = trapezoidal_membership(data[col], 3.3, 3.75, 4.0, 5.0)

    # --- A-Level (Separate, but handled carefully) ---
    # We MASK zeros so they don't count as "Low"
    # alevel_mask = (data['A-Level (As)'] > 0).astype(int)
    data['Fuzzy_ALevel_Low'] = triangular_membership(data['A-Level (As)'], -1, 0, 2) #Min: 2
    data['Fuzzy_ALevel_Med'] = triangular_membership(data['A-Level (As)'], 1, 2, 3) #Med: 3
    data['Fuzzy_ALevel_High'] = trapezoidal_membership(data['A-Level (As)'], 2, 3, 4, 5) #Max: 4
    
    # --- Co-curricular Score ---
    data['Fuzzy_CoCurr_Low'] = triangular_membership(data['Co-curricular Score'], -1, 0, 50) #Min: 30
    data['Fuzzy_CoCurr_Med'] = triangular_membership(data['Co-curricular Score'], 40, 60, 80) #Med: 50
    data['Fuzzy_CoCurr_High'] = trapezoidal_membership(data['Co-curricular Score'], 70, 90, 100, 120) #Max: 100
    
    return data

def add_engineered_features(df):
    """Add additional engineered features for better model performance."""
    data = df.copy()
    
    # Ensure numeric columns are properly typed to avoid string arithmetic issues
    if 'Household Income' in data.columns:
        data['Household Income'] = pd.to_numeric(data['Household Income'], errors='coerce').fillna(0.0)
    if 'Household Income (Max Annual RM)' in data.columns:
        data['Household Income (Max Annual RM)'] = pd.to_numeric(
            data['Household Income (Max Annual RM)'], errors='coerce').fillna(0.0)

    # Academic Performance Score (weighted combination)
    data['Academic_Score'] = (
        data['SPM Result (As)'] * 0.3 +
        data['CGPA_Unified'] * 0.5 +
        data['A-Level (As)'] * 0.2
    )
    
    # Overall Performance Score
    data['Overall_Score'] = (
        data['Academic_Score'] * 0.7 +
        data['Co-curricular Score'] * 0.3
    )
    
    # Income to Scholarship Ratio (if scholarship max income exists)
    if 'Household Income (Max Annual RM)' in data.columns:
        # Guard against division by zero and non-numeric types
        try:
            denom = data['Household Income (Max Annual RM)'].replace(0, np.nan)
            # Add 1 to avoid zero-division where denom was zero; fillna to handle original zeros
            data['Income_Ratio'] = data['Household Income'] / (denom + 1).fillna(1)
            # Where denom was originally zero, fallback to Household Income (so ratio ~= income)
            data['Income_Ratio'] = data['Income_Ratio'].fillna(data['Household Income'])
        except Exception:
            # Fallback: set ratio to 0 to avoid breaking pipeline
            data['Income_Ratio'] = 0.0
    
    # Age Group Categories
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 18, 21, 25, 100], labels=['<18', '18-21', '21-25', '25+'])
    
    # Has Academic Qualification (binary)
    data['Has_CGPA'] = (data['CGPA_Unified'] > 0).astype(int)
    data['Has_ALevel'] = (data['A-Level (As)'] > 0).astype(int)
    
    # Academic Excellence (top tier performance)
    data['Is_High_Performer'] = (
        ((data['SPM Result (As)'] >= 8) | (data['CGPA_Unified'] >= 3.5) | (data['A-Level (As)'] >= 3)) &
        (data['Co-curricular Score'] >= 60)
    ).astype(int)
    
    # Financial Need (low income indicator)
    data['High_Financial_Need'] = (data['Household Income'] < 24000).astype(int)
    
    return data

# ==========================================
# 2. Data Loading & Cleaning
# ==========================================
def clean_currency(x):
    if isinstance(x, str):
        clean_str = x.replace('RM', '').replace(',', '').strip()
        try:
            return float(clean_str)
        except ValueError:
            return 0.0
    return x if pd.notnull(x) else 0.0

def load_data(student_path, scholarship_path):
    df_students = pd.read_csv(student_path)
    df_scholarships = pd.read_csv(scholarship_path)

    # Cleaning
    df_students['Household Income'] = df_students['Household Income'].apply(clean_currency)
    df_scholarships['Household Income (Max Annual RM)'] = df_scholarships['Household Income (Max Annual RM)'].apply(clean_currency)
    df_students['Applied Scholarship'] = df_students['Applied Scholarship'].str.strip()
    df_scholarships['Scholarship Name'] = df_scholarships['Scholarship Name'].str.strip()
    df_scholarships = df_scholarships.rename(columns={'UG/PG': 'Scholarship Level'})

    # Handle Missing Numeric Cols
    numeric_cols = ['SPM Result (As)', 'STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA', 'A-Level (As)']
    for col in numeric_cols:
        if col in df_students.columns:
            df_students[col] = df_students[col].fillna(0.0)
        else:
            df_students[col] = 0.0

    # Create Unified CGPA (The Robust Feature)
    # Takes the first non-null/non-zero value from the hierarchy or max
    cgpa_sources = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
    # We use max() to pick the best score the student has
    df_students['CGPA_Unified'] = df_students[cgpa_sources].max(axis=1)
    
    # Targets - Binary for classification
    df_students['Target_Eligibility'] = df_students['Eligibility Status'].apply(lambda x: 1 if str(x).strip() == 'Eligible' else 0)
    df_students['Target_Acceptance'] = df_students['Application Status'].apply(lambda x: 1 if str(x).strip() == 'Accepted' else 0)
    
    # Target - Continuous Success Probability Score (0-100)
    # Combined score: Eligibility (40%) + Acceptance (60%)
    df_students['Success_Probability'] = (
        df_students['Target_Eligibility'] * 40 + 
        df_students['Target_Acceptance'] * 60
    ).astype(float)

    return df_students, df_scholarships

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Loading data and training models...")
    df_students, df_scholarships = load_data('../dataset/FINAL_STUDENTS.csv', '../dataset/SCHOLARSHIPS_2.csv')
    
    # Merge for Training
    df_train = pd.merge(df_students, df_scholarships, 
                        left_on=['Applied Scholarship', 'Scholarship Level'], 
                        right_on=['Scholarship Name', 'Scholarship Level'], 
                        how='left')
    
    # Ensure Unified CGPA exists in merged (it comes from students, so it should be there)
    # Re-apply Max Logic just in case of merge artifacts (redundant but safe)
    cgpa_sources = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
    df_train['CGPA_Unified'] = df_train[cgpa_sources].max(axis=1)
    

    #Finalized df before training
    df_train = apply_fuzzy_logic(df_train)
    df_train = add_engineered_features(df_train)

    # --- Feature Selection ---
    categorical_features = ['Field of Study', 'Race', 'Applied Scholarship', 'Scholarship Level', 'Age_Group']
    
    # Model 1 (Fuzzy + GB) Features
    fuzzy_features = [col for col in df_train.columns if col.startswith('Fuzzy_')]
    engineered_features_m1 = ['Academic_Score', 'Overall_Score', 'Has_CGPA', 'Has_ALevel', 
                               'Is_High_Performer', 'High_Financial_Need']
    if 'Income_Ratio' in df_train.columns:
        engineered_features_m1.append('Income_Ratio')
    numeric_features_m1 = ['Age', 'Household Income'] + engineered_features_m1
    
    # Model 2 (Elastic Net) Features
    # Note: We use Unified CGPA here too for robustness
    engineered_features_m2 = ['Academic_Score', 'Overall_Score', 'Has_CGPA', 'Has_ALevel', 
                               'Is_High_Performer', 'High_Financial_Need']
    if 'Income_Ratio' in df_train.columns:
        engineered_features_m2.append('Income_Ratio')
    numeric_features_m2 = ['Age', 'Household Income', 'SPM Result (As)', 'CGPA_Unified', 
                           'A-Level (As)', 'Co-curricular Score'] + engineered_features_m2

    X = df_train
    y_success = df_train['Success_Probability']  # Continuous target for regression
    y_eligibility = df_train['Target_Eligibility']  # Binary target for classification metrics
    
    print(f"\n=== Dataset Information ===")
    print(f"Total samples: {len(X)}")
    print(f"Success Probability - Mean: {y_success.mean():.2f}, Std: {y_success.std():.2f}")
    print(f"Eligibility Rate: {y_eligibility.mean():.2%}")

    # --- Train-Test Split for Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_success, test_size=0.2, random_state=42
    )
    # Get corresponding eligibility labels for classification metrics
    y_elig_train = X_train['Target_Eligibility']
    y_elig_test = X_test['Target_Eligibility']
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # ==========================================
    # MODEL 1: HYBRID MODEL (Fuzzy Logic + Gradient Boosting Regressor)
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING MODEL 1: HYBRID (Fuzzy Logic + Gradient Boosting)")
    print("="*60)
    
    preprocessor_hybrid = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features_m1),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('fuzzy', 'passthrough', fuzzy_features)])
    
    hybrid_pipe = Pipeline([
        ('preprocessor', preprocessor_hybrid), 
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])

    # Hyperparameter tuning for Hybrid Model
    hybrid_params = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 4, 5],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__subsample': [0.8, 0.9, 1.0]
    }
    
    print("\nPerforming Grid Search with 5-Fold Cross-Validation...")
    hybrid_model = GridSearchCV(
        hybrid_pipe, 
        hybrid_params, 
        cv=5, 
        scoring='neg_mean_squared_error',
        n_jobs=-1, 
        verbose=1
    )
    hybrid_model.fit(X_train, y_train)
    
    print(f"\n✓ Best Hybrid Model Parameters:")
    for param, value in hybrid_model.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"✓ Best CV RMSE: {np.sqrt(-hybrid_model.best_score_):.4f}")

    # ==========================================
    # MODEL 2: BASELINE MODEL (Elastic Net Regression)
    # ==========================================
    print("\n" + "="*60)
    print("TRAINING MODEL 2: BASELINE (Elastic Net Regression)")
    print("="*60)
    
    preprocessor_baseline = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features_m2),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])
    
    baseline_pipe = Pipeline([
        ('preprocessor', preprocessor_baseline),
        ('regressor', ElasticNet(random_state=42, max_iter=5000))
    ])

    # Hyperparameter tuning for Baseline Model
    baseline_params = {
        'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    print("\nPerforming Grid Search with 5-Fold Cross-Validation...")
    baseline_model = GridSearchCV(
        baseline_pipe,
        baseline_params,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    baseline_model.fit(X_train, y_train)
    
    print(f"\n✓ Best Baseline Model Parameters:")
    for param, value in baseline_model.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"✓ Best CV RMSE: {np.sqrt(-baseline_model.best_score_):.4f}")

    # ==========================================
    # MODEL EVALUATION & COMPARISON
    # ==========================================
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Predictions
    y_pred_hybrid = hybrid_model.predict(X_test)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Clip predictions to valid range [0, 100]
    y_pred_hybrid = np.clip(y_pred_hybrid, 0, 100)
    y_pred_baseline = np.clip(y_pred_baseline, 0, 100)
    
    # Convert to binary for classification metrics (threshold: 40)
    y_pred_hybrid_binary = (y_pred_hybrid >= 40).astype(int)
    y_pred_baseline_binary = (y_pred_baseline >= 40).astype(int)
    
    # ========== REGRESSION METRICS ==========
    print("\n" + "-"*60)
    print("REGRESSION METRICS (Success Probability Prediction)")
    print("-"*60)
    
    # Hybrid Model Regression Metrics
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
    hybrid_mae = mean_absolute_error(y_test, y_pred_hybrid)
    hybrid_r2 = r2_score(y_test, y_pred_hybrid)
    
    print("\n📊 MODEL 1 - HYBRID (Fuzzy Logic + Gradient Boosting):")
    print(f"  RMSE: {hybrid_rmse:.4f}")
    print(f"  MAE:  {hybrid_mae:.4f}")
    print(f"  R²:   {hybrid_r2:.4f}")
    
    # Baseline Model Regression Metrics
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
    baseline_r2 = r2_score(y_test, y_pred_baseline)
    
    print("\n📊 MODEL 2 - BASELINE (Elastic Net):")
    print(f"  RMSE: {baseline_rmse:.4f}")
    print(f"  MAE:  {baseline_mae:.4f}")
    print(f"  R²:   {baseline_r2:.4f}")
    
    # ========== CLASSIFICATION METRICS ==========
    print("\n" + "-"*60)
    print("CLASSIFICATION METRICS (Eligibility Matching)")
    print("-"*60)
    
    # Hybrid Model Classification Metrics
    hybrid_f1 = f1_score(y_elig_test, y_pred_hybrid_binary)
    hybrid_precision = precision_score(y_elig_test, y_pred_hybrid_binary)
    hybrid_recall = recall_score(y_elig_test, y_pred_hybrid_binary)
    hybrid_accuracy = accuracy_score(y_elig_test, y_pred_hybrid_binary)
    
    print("\n🎯 MODEL 1 - HYBRID (Fuzzy Logic + Gradient Boosting):")
    print(f"  F1 Score:  {hybrid_f1:.4f}")
    print(f"  Precision: {hybrid_precision:.4f}")
    print(f"  Recall:    {hybrid_recall:.4f}")
    print(f"  Accuracy:  {hybrid_accuracy:.4f}")
    
    # Baseline Model Classification Metrics
    baseline_f1 = f1_score(y_elig_test, y_pred_baseline_binary)
    baseline_precision = precision_score(y_elig_test, y_pred_baseline_binary)
    baseline_recall = recall_score(y_elig_test, y_pred_baseline_binary)
    baseline_accuracy = accuracy_score(y_elig_test, y_pred_baseline_binary)
    
    print("\n🎯 MODEL 2 - BASELINE (Elastic Net):")
    print(f"  F1 Score:  {baseline_f1:.4f}")
    print(f"  Precision: {baseline_precision:.4f}")
    print(f"  Recall:    {baseline_recall:.4f}")
    print(f"  Accuracy:  {baseline_accuracy:.4f}")
    
    # ========== SUMMARY COMPARISON ==========
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    print("\n📈 Regression Performance (Lower is Better for RMSE/MAE):")
    print(f"  {'Metric':<15} {'Hybrid':<15} {'Baseline':<15} {'Winner':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'RMSE':<15} {hybrid_rmse:<15.4f} {baseline_rmse:<15.4f} {'Hybrid' if hybrid_rmse < baseline_rmse else 'Baseline':<15}")
    print(f"  {'MAE':<15} {hybrid_mae:<15.4f} {baseline_mae:<15.4f} {'Hybrid' if hybrid_mae < baseline_mae else 'Baseline':<15}")
    print(f"  {'R²':<15} {hybrid_r2:<15.4f} {baseline_r2:<15.4f} {'Hybrid' if hybrid_r2 > baseline_r2 else 'Baseline':<15}")
    
    print("\n🎯 Classification Performance (Higher is Better):")
    print(f"  {'Metric':<15} {'Hybrid':<15} {'Baseline':<15} {'Winner':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"  {'F1 Score':<15} {hybrid_f1:<15.4f} {baseline_f1:<15.4f} {'Hybrid' if hybrid_f1 > baseline_f1 else 'Baseline':<15}")
    print(f"  {'Precision':<15} {hybrid_precision:<15.4f} {baseline_precision:<15.4f} {'Hybrid' if hybrid_precision > baseline_precision else 'Baseline':<15}")
    print(f"  {'Recall':<15} {hybrid_recall:<15.4f} {baseline_recall:<15.4f} {'Hybrid' if hybrid_recall > baseline_recall else 'Baseline':<15}")
    print(f"  {'Accuracy':<15} {hybrid_accuracy:<15.4f} {baseline_accuracy:<15.4f} {'Hybrid' if hybrid_accuracy > baseline_accuracy else 'Baseline':<15}")
    
    # ========== SAVE MODELS ==========
    print("\n" + "="*60)
    print("SAVING TRAINED MODELS")
    print("="*60)
    
    joblib.dump(hybrid_model, 'hybrid_model.pkl')
    joblib.dump(baseline_model, 'baseline_model.pkl')
    
    print("✓ Hybrid Model saved as: hybrid_model.pkl")
    print("✓ Baseline Model saved as: baseline_model.pkl")
    
    print("\n=== Training Complete ===\n")

    # ======================================================
    # TEST STUDENT DETAILS
    # ======================================================
    student_profile = {
        'Age': 18,
        'Race': 'Malay',
        'Household Income': 2000.0,
        'SPM Result (As)': 9,            # Excellent SPM
        'Co-curricular Score': 100,       # Excellent Co-Curr
        'Field of Study': 'Engineering',
        
        # Scenario: Perfect Matriculation, No STPM/A-Levels
        'STPM CGPA': 0.0,
        'Matriculation CGPA': 0.0,       # Perfect Score
        'Foundation CGPA': 0.0,
        'A-Level (As)': 0.0,
        'UG CGPA': 0.0                   # Not yet graduated
    }
    # ======================================================
    
    print(f"Predicting for student: {student_profile}")

    unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
    test_df = pd.DataFrame([student_profile] * len(unique_scholarships))
    
    # Calculate Unified CGPA for Test Data
    test_df['CGPA_Unified'] = test_df[cgpa_sources].max(axis=1)
    
    test_df['Applied Scholarship'] = unique_scholarships['Scholarship Name'].values
    test_df['Scholarship Level'] = unique_scholarships['Scholarship Level'].values

    test_merged = pd.merge(test_df, df_scholarships, 
                           left_on=['Applied Scholarship', 'Scholarship Level'], 
                           right_on=['Scholarship Name', 'Scholarship Level'], 
                           how='left')
    
    test_processed = apply_fuzzy_logic(test_merged)
    test_processed = add_engineered_features(test_processed)

    # ==========================================
    # AUTOMATED SCHOLARSHIP MATCHING
    # ==========================================
    print("\n" + "="*60)
    print("AUTOMATED SCHOLARSHIP MATCHING FOR TEST STUDENT")
    print("="*60)
    
    # Predict Success Probability with both models
    success_prob_hybrid = hybrid_model.predict(test_processed)
    success_prob_baseline = baseline_model.predict(test_processed)
    
    # Clip to valid range [0, 100]
    success_prob_hybrid = np.clip(success_prob_hybrid, 0, 100)
    success_prob_baseline = np.clip(success_prob_baseline, 0, 100)
    
    # Determine Eligibility (threshold: 40)
    eligibility_hybrid = (success_prob_hybrid >= 40).astype(int)
    eligibility_baseline = (success_prob_baseline >= 40).astype(int)

    # --- RESULT GENERATION WITH HARD RULES ---
    final_results = []
    bumiputera_races = ['Malay', 'Bumiputera Sabah / Sarawak'] 

    for i, row in unique_scholarships.iterrows():
        scholarship_name = row['Scholarship Name']
        scholarship_level = row['Scholarship Level']
        
        # --- HARD RULES CHECKING ---
        # 1. Bumiputera Rule
        rule_row = df_scholarships[(df_scholarships['Scholarship Name'] == scholarship_name) & 
                                   (df_scholarships['Scholarship Level'] == scholarship_level)].iloc[0]
        is_bumi_only = rule_row['Bumiputera Only'] == 'Yes'
        
        # 2. PG Rule (Must have UG CGPA > 0)
        pg_requirement_fail = False
        if scholarship_level == 'PG':
            if student_profile['UG CGPA'] <= 0:
                pg_requirement_fail = True

        # Check Disqualification
        disqualified = False
        disqualification_reason = []
        if is_bumi_only and (student_profile['Race'] not in bumiputera_races):
            disqualified = True
            disqualification_reason.append("Not Bumiputera")
        if pg_requirement_fail:
            disqualified = True
            disqualification_reason.append("No UG degree for PG scholarship")

        # Apply Hard Rules to Model Predictions
        is_eligible_hybrid = (eligibility_hybrid[i] == 1) and (not disqualified)
        is_eligible_baseline = (eligibility_baseline[i] == 1) and (not disqualified)
        
        # Only include scholarships where at least one model predicts eligibility
        if is_eligible_hybrid or is_eligible_baseline:
            result = {
                "Scholarship": scholarship_name,
                "Level": scholarship_level,
                "Hybrid_Model": {
                    "Eligibility": "Eligible" if is_eligible_hybrid else "Ineligible",
                    "Success_Probability": f"{success_prob_hybrid[i]:.2f}%",
                    "Raw_Score": float(success_prob_hybrid[i])
                },
                "Baseline_Model": {
                    "Eligibility": "Eligible" if is_eligible_baseline else "Ineligible",
                    "Success_Probability": f"{success_prob_baseline[i]:.2f}%",
                    "Raw_Score": float(success_prob_baseline[i])
                },
                "Recommendation": "Highly Recommended" if (success_prob_hybrid[i] >= 70 or success_prob_baseline[i] >= 70) else "Recommended"
            }
            
            if disqualified:
                result["Note"] = f"Disqualified: {', '.join(disqualification_reason)}"
            
            final_results.append(result)
    
    # Sort by average success probability (descending)
    final_results = sorted(
        final_results, 
        key=lambda x: (x['Hybrid_Model']['Raw_Score'] + x['Baseline_Model']['Raw_Score']) / 2, 
        reverse=True
    )
    
    # Display Results
    print(f"\n✓ Found {len(final_results)} matching scholarships for the student\n")
    print(json.dumps(final_results, indent=2))
    
    # Save to file
    # output_filename = 'matched_scholarships.json'
    # with open(output_filename, 'w') as f:
    #     json.dump(final_results, f, indent=2)
    # print(f"\n✓ Results saved to '{output_filename}'")
    
    # ==========================================
    # OBJECTIVES ACHIEVEMENT SUMMARY
    # ==========================================
    print("\n" + "="*60)
    print("OBJECTIVES ACHIEVEMENT SUMMARY")
    print("="*60)
    
    print("\n✅ OBJECTIVE 1: Automated Scholarship Matching")
    print("   Status: ACHIEVED")
    print("   - System analyzes student's academic results automatically")
    print("   - Identifies eligible scholarships based on:")
    print("     • SPM Results, CGPA, A-Level scores")
    print("     • Co-curricular activities")
    print("     • Household income")
    print("     • Race and scholarship requirements")
    print(f"   - Matched {len(final_results)} scholarships for test student")
    
    print("\n✅ OBJECTIVE 2: Scholarship Success Probability Prediction")
    print("   Status: ACHIEVED")
    print("   - AI models estimate success likelihood (0-100%)")
    print("   - Models trained and evaluated:")
    print(f"     • Hybrid Model - RMSE: {hybrid_rmse:.4f}, F1: {hybrid_f1:.4f}")
    print(f"     • Baseline Model - RMSE: {baseline_rmse:.4f}, F1: {baseline_f1:.4f}")
    
    print("\n📊 MODEL COMPARISON RESULTS:")
    print("   Regression Metrics (RMSE/MAE):")
    if hybrid_rmse < baseline_rmse and hybrid_mae < baseline_mae:
        print("   ✓ Winner: HYBRID MODEL (Fuzzy Logic + Gradient Boosting)")
        print("     - Better prediction accuracy for success probability")
    elif baseline_rmse < hybrid_rmse and baseline_mae < hybrid_mae:
        print("   ✓ Winner: BASELINE MODEL (Elastic Net)")
        print("     - Better prediction accuracy for success probability")
    else:
        print("   ✓ Mixed Results - Each model has strengths")
    
    print("\n   Classification Metrics (F1 Score):")
    if hybrid_f1 > baseline_f1:
        print("   ✓ Winner: HYBRID MODEL (Fuzzy Logic + Gradient Boosting)")
        print("     - Better at identifying eligible scholarships")
    elif baseline_f1 > hybrid_f1:
        print("   ✓ Winner: BASELINE MODEL (Elastic Net)")
        print("     - Better at identifying eligible scholarships")
    else:
        print("   ✓ Both models perform equally well")
    
    print("\n" + "="*60)
    print("ALL OBJECTIVES SUCCESSFULLY ACHIEVED!")
    print("="*60)