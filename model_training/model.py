import pandas as pd
import numpy as np
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Fuzzy Logic Functions
# ==========================================
def triangular_membership(x, a, b, c):
    """Buat triangular membership function"""
    term1 = (x - a) / (b - a + 1e-6)
    term2 = (c - x) / (c - b + 1e-6)
    return np.maximum(0, np.minimum(term1, term2))

def trapezoidal_membership(x, a, b, c, d):
    """Buat trapezoidal membership function"""
    term1 = (x - a) / (b - a + 1e-6)
    term2 = (d - x) / (d - c + 1e-6)
    return np.maximum(0, np.minimum(np.minimum(term1, 1), term2))

def apply_fuzzy_logic(df):
    """Applies fuzzy logic to numerical columns"""
    data = df.copy()
    
    # Income Fuzzy Sets
    data['Fuzzy_Income_Low'] = triangular_membership(data['Household Income'], 0, 0, 12000)
    data['Fuzzy_Income_Med'] = triangular_membership(data['Household Income'], 40000, 75000, 110000)
    data['Fuzzy_Income_High'] = trapezoidal_membership(data['Household Income'], 90000, 150000, 350000, 350000)
    
    # SPM Fuzzy Sets
    data['Fuzzy_SPM_Low'] = triangular_membership(data['SPM Result (As)'], 0, 1, 3)
    data['Fuzzy_SPM_Med'] = triangular_membership(data['SPM Result (As)'], 3, 5, 7)
    data['Fuzzy_SPM_High'] = trapezoidal_membership(data['SPM Result (As)'], 7, 10, 20, )
    
    # CGPA Fuzzy Sets
    data['Fuzzy_CGPA_Low'] = triangular_membership(data['CGPA_Unified'], 0.0, 2.0, 2.8)
    data['Fuzzy_CGPA_Med'] = triangular_membership(data['CGPA_Unified'], 2.5, 3.0, 3.5)
    data['Fuzzy_CGPA_High'] = trapezoidal_membership(data['CGPA_Unified'], 3.3, 3.7, 4.0, 4.0)
    
    # Co-Curricular Fuzzy Sets
    data['Fuzzy_CoCurr_Low'] = triangular_membership(data['Co-curricular Score'], 0, 20, 50)
    data['Fuzzy_CoCurr_Med'] = triangular_membership(data['Co-curricular Score'], 40, 60, 80)
    data['Fuzzy_CoCurr_High'] = trapezoidal_membership(data['Co-curricular Score'], 70, 90, 100, 100)
    
    return data

# ==========================================
# 2. Data Loading & Helper
# ==========================================
def clean_currency(x):
    """Cleans currency strings."""
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

    # Unified CGPA
    academic_cols = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
    df_students['CGPA_Unified'] = df_students[academic_cols].bfill(axis=1).iloc[:, 0].fillna(0)

    # Targets
    df_students['Target_Eligibility'] = df_students['Eligibility Status'].apply(lambda x: 1 if str(x).strip() == 'Eligible' else 0)
    df_students['Target_Acceptance'] = df_students['Application Status'].apply(lambda x: 1 if str(x).strip() == 'Accepted' else 0)

    return df_students, df_scholarships

# ==========================================
# 3. Main Logic: Train & Analyze
# ==========================================
if __name__ == "__main__":
    # --- A. Load & Prepare ---
    df_students, df_scholarships = load_data('FINAL_STUDENTS.csv', 'SCHOLARSHIPS.csv')
    
    # Merge for Training Context
    df_train = pd.merge(df_students, df_scholarships, 
                        left_on=['Applied Scholarship', 'Scholarship Level'], 
                        right_on=['Scholarship Name', 'Scholarship Level'], 
                        how='left')
    df_train = apply_fuzzy_logic(df_train)

    # Select Features
    fuzzy_features = [col for col in df_train.columns if col.startswith('Fuzzy_')]
    categorical_features = ['Field of Study', 'Race', 'Applied Scholarship']
    numeric_features = ['Age', 'Household Income']
    features = fuzzy_features + categorical_features + numeric_features

    X = df_train[features]
    y_eligibility = df_train['Target_Eligibility']
    y_acceptance = df_train['Target_Acceptance']

    # Split Data
    X_train, X_test, y_elig_train, y_elig_test, y_acc_train, y_acc_test = train_test_split(
        X, y_eligibility, y_acceptance, test_size=0.2, random_state=42
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('fuzzy', 'passthrough', fuzzy_features)
        ])

    # --- B. Define & Train Models ---
    print("Training Models...")

    # Model 1: Fuzzy + Gradient Boosting
    gb_elig = Pipeline([('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))])
    gb_acc = Pipeline([('preprocessor', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))])

    # Model 2: Fuzzy + Elastic Net (using SGDClassifier with elasticnet penalty)
    en_elig = Pipeline([('preprocessor', preprocessor), ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=42))])
    en_acc = Pipeline([('preprocessor', preprocessor), ('clf', SGDClassifier(loss='log_loss', penalty='elasticnet', l1_ratio=0.5, random_state=42))])

    # Fit
    gb_elig.fit(X_train, y_elig_train)
    gb_acc.fit(X_train, y_acc_train)
    en_elig.fit(X_train, y_elig_train)
    en_acc.fit(X_train, y_acc_train)

    # --- C. Analyze Metrics ---
    def get_metrics(name, m_elig, m_acc, X_t, y_e_t, y_a_t):
        pred_elig = m_elig.predict(X_t)
        prob_acc = m_acc.predict_proba(X_t)[:, 1]
        return {
            'Model': name,
            'Eligibility F1': f1_score(y_e_t, pred_elig),
            'Acceptance RMSE': np.sqrt(mean_squared_error(y_a_t, prob_acc)),
            'Acceptance MAE': mean_absolute_error(y_a_t, prob_acc)
        }

    met_gb = get_metrics("Fuzzy + Gradient Boosting", gb_elig, gb_acc, X_test, y_elig_test, y_acc_test)
    met_en = get_metrics("Fuzzy + Elastic Net", en_elig, en_acc, X_test, y_elig_test, y_acc_test)
    
    print("\n--- Model Comparison Metrics ---")
    print(pd.DataFrame([met_gb, met_en]))

    # --- D. Predict for Sample Student ---
    # Sample Input (From Dataset Row 0)
    sample_student = df_students.iloc[0].copy()
    
    # Expand to all scholarships
    unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
    test_df = pd.DataFrame([sample_student] * len(unique_scholarships))
    test_df['Applied Scholarship'] = unique_scholarships['Scholarship Name'].values
    test_df['Scholarship Level'] = unique_scholarships['Scholarship Level'].values

    # Merge & Process
    test_merged = pd.merge(test_df, df_scholarships, 
                           left_on=['Applied Scholarship', 'Scholarship Level'], 
                           right_on=['Scholarship Name', 'Scholarship Level'], 
                           how='left')
    test_processed = apply_fuzzy_logic(test_merged)
    X_batch = test_processed[features]

    # Predict with Both Models
    # GB
    res_gb_elig = gb_elig.predict(X_batch)
    res_gb_prob = gb_acc.predict_proba(X_batch)[:, 1]
    
    # EN
    res_en_elig = en_elig.predict(X_batch)
    res_en_prob = en_acc.predict_proba(X_batch)[:, 1]

    # Format Output
    final_results = []
    for i, row in unique_scholarships.iterrows():
        # Model 1 Output
        elig_1 = "Eligible" if res_gb_elig[i] == 1 else "Ineligible"
        prob_1 = f"{res_gb_prob[i]:.1%}" if res_gb_elig[i] == 1 else "N/A"
        
        # Model 2 Output
        elig_2 = "Eligible" if res_en_elig[i] == 1 else "Ineligible"
        prob_2 = f"{res_en_prob[i]:.1%}" if res_en_elig[i] == 1 else "N/A"
        
        final_results.append({
            "Scholarship": row['Scholarship Name'],
            "Level": row['Scholarship Level'],
            "Model 1 (GB) Eligibility": elig_1,
            "Model 1 (GB) Probability": prob_1,
            "Model 2 (EN) Eligibility": elig_2,
            "Model 2 (EN) Probability": prob_2
        })

    # Save JSON
    with open('model_comparison_output.json', 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print("\nPrediction JSON Saved to 'model_comparison_output.json'")
