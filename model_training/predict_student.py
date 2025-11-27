import pandas as pd
import numpy as np
import joblib
import json
from gemini_train import (
    apply_fuzzy_logic, 
    add_engineered_features,
    load_data,
    clean_currency
)

#Predict gune salah satu model (hybrid/baseline)
def predict_scholarship_match(student_profile, model_path='hybrid_model.pkl', use_hybrid=True):
    """
    Load trained model and predict scholarship success probability for a student.
    
    Args:
        student_profile: Dictionary containing student information
        model_path: Path to the saved model ('hybrid_model.pkl' or 'baseline_model.pkl')
        use_hybrid: Boolean to choose between hybrid (True) or baseline (False) model
    
    Returns:
        List of matched scholarships with success probabilities
    """
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    if use_hybrid:
        model = joblib.load('hybrid_model.pkl')
        model_name = "Hybrid (Fuzzy Logic + Gradient Boosting)"
    else:
        model = joblib.load('baseline_model.pkl')
        model_name = "Baseline (Elastic Net)"
    print(f"✓ {model_name} model loaded successfully\n")
    
    # Load scholarship data
    _, df_scholarships = load_data('../dataset/FINAL_STUDENTS.csv', '../dataset/SCHOLARSHIPS.csv')
    
    # Get all unique scholarships
    unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
    
    # Create test dataframe with student profile repeated for each scholarship
    test_df = pd.DataFrame([student_profile] * len(unique_scholarships))
    
    # Calculate Unified CGPA
    cgpa_sources = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
    test_df['CGPA_Unified'] = test_df[cgpa_sources].max(axis=1)
    
    # Add scholarship information
    test_df['Applied Scholarship'] = unique_scholarships['Scholarship Name'].values
    test_df['Scholarship Level'] = unique_scholarships['Scholarship Level'].values
    
    # Merge with scholarship details
    test_merged = pd.merge(
        test_df, 
        df_scholarships,
        left_on=['Applied Scholarship', 'Scholarship Level'],
        right_on=['Scholarship Name', 'Scholarship Level'],
        how='left'
    )
    
    # Apply feature engineering
    test_processed = apply_fuzzy_logic(test_merged)
    test_processed = add_engineered_features(test_processed)
    
    # Predict success probability
    print("Predicting scholarship success probabilities...")
    success_probabilities = model.predict(test_processed)
    success_probabilities = np.clip(success_probabilities, 0, 100)
    
    # Determine eligibility (threshold: 40)
    eligibility = (success_probabilities >= 40).astype(int)
    
    # Generate results with hard rules
    final_results = []
    bumiputera_races = ['Malay', 'Bumiputera Sabah / Sarawak']
    
    for i, row in unique_scholarships.iterrows():
        scholarship_name = row['Scholarship Name']
        scholarship_level = row['Scholarship Level']
        
        # Get scholarship rules
        rule_row = df_scholarships[
            (df_scholarships['Scholarship Name'] == scholarship_name) & 
            (df_scholarships['Scholarship Level'] == scholarship_level)
        ].iloc[0]
        
        is_bumi_only = rule_row['Bumiputera Only'] == 'Yes'
        
        # Check hard rules
        disqualified = False
        disqualification_reasons = []
        
        # Rule 1: Bumiputera requirement
        if is_bumi_only and (student_profile['Race'] not in bumiputera_races):
            disqualified = True
            disqualification_reasons.append("Not Bumiputera")
        
        # Rule 2: PG scholarship requires UG degree
        if scholarship_level == 'PG' and student_profile['UG CGPA'] <= 0:
            disqualified = True
            disqualification_reasons.append("No UG degree for PG scholarship")
        
        # Apply eligibility
        is_eligible = (eligibility[i] == 1) and (not disqualified)
        
        if is_eligible:
            result = {
                "Scholarship": scholarship_name,
                "Level": scholarship_level,
                "Success_Probability": f"{success_probabilities[i]:.2f}%",
                "Raw_Score": float(success_probabilities[i]),
                "Eligibility": "Eligible",
                "Recommendation": "Highly Recommended" if success_probabilities[i] >= 70 else "Recommended",
                "Model_Used": model_name
            }
            final_results.append(result)
        elif not disqualified and eligibility[i] == 0:
            # Low probability but not disqualified
            result = {
                "Scholarship": scholarship_name,
                "Level": scholarship_level,
                "Success_Probability": f"{success_probabilities[i]:.2f}%",
                "Raw_Score": float(success_probabilities[i]),
                "Eligibility": "Low Probability",
                "Recommendation": "Not Recommended",
                "Model_Used": model_name
            }
    
    # Sort by success probability (descending)
    final_results = sorted(final_results, key=lambda x: x['Raw_Score'], reverse=True)
    
    return final_results

#Predict guna dua dua model
def predict_with_both_models(student_profile):
    """
    Predict using both Hybrid and Baseline models for comparison.
    
    Args:
        student_profile: Dictionary containing student information
    
    Returns:
        List of matched scholarships with predictions from both models
    """
    
    print("="*60)
    print("LOADING BOTH MODELS FOR COMPARISON")
    print("="*60 + "\n")
    
    # Load both models
    hybrid_model = joblib.load('hybrid_model.pkl')
    baseline_model = joblib.load('baseline_model.pkl')
    print("✓ Both models loaded successfully\n")
    
    # Load scholarship data
    _, df_scholarships = load_data('../dataset/FINAL_STUDENTS.csv', '../dataset/SCHOLARSHIPS.csv')
    
    # Get all unique scholarships
    unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
    
    # Create test dataframe
    test_df = pd.DataFrame([student_profile] * len(unique_scholarships))
    
    # Calculate Unified CGPA
    cgpa_sources = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
    test_df['CGPA_Unified'] = test_df[cgpa_sources].max(axis=1)
    
    # Add scholarship information
    test_df['Applied Scholarship'] = unique_scholarships['Scholarship Name'].values
    test_df['Scholarship Level'] = unique_scholarships['Scholarship Level'].values
    
    # Merge with scholarship details
    test_merged = pd.merge(
        test_df, 
        df_scholarships,
        left_on=['Applied Scholarship', 'Scholarship Level'],
        right_on=['Scholarship Name', 'Scholarship Level'],
        how='left'
    )
    
    # Apply feature engineering
    test_processed = apply_fuzzy_logic(test_merged)
    test_processed = add_engineered_features(test_processed)
    
    # Predict with both models
    print("Predicting with both models...")
    success_prob_hybrid = np.clip(hybrid_model.predict(test_processed), 0, 100)
    success_prob_baseline = np.clip(baseline_model.predict(test_processed), 0, 100)
    
    # Determine eligibility (threshold: 40)
    eligibility_hybrid = (success_prob_hybrid >= 40).astype(int)
    eligibility_baseline = (success_prob_baseline >= 40).astype(int)
    
    # Generate results
    final_results = []
    bumiputera_races = ['Malay', 'Bumiputera Sabah / Sarawak']
    
    for i, row in unique_scholarships.iterrows():
        scholarship_name = row['Scholarship Name']
        scholarship_level = row['Scholarship Level']
        
        # Get scholarship rules
        rule_row = df_scholarships[
            (df_scholarships['Scholarship Name'] == scholarship_name) & 
            (df_scholarships['Scholarship Level'] == scholarship_level)
        ].iloc[0]
        
        is_bumi_only = rule_row['Bumiputera Only'] == 'Yes'
        
        # Check hard rules
        disqualified = False
        disqualification_reasons = []
        
        if is_bumi_only and (student_profile['Race'] not in bumiputera_races):
            disqualified = True
            disqualification_reasons.append("Not Bumiputera")
        
        if scholarship_level == 'PG' and student_profile['UG CGPA'] <= 0:
            disqualified = True
            disqualification_reasons.append("No UG degree for PG scholarship")
        
        # Apply eligibility for both models
        is_eligible_hybrid = (eligibility_hybrid[i] == 1) and (not disqualified)
        is_eligible_baseline = (eligibility_baseline[i] == 1) and (not disqualified)
        
        # Include if either model predicts eligibility
        if is_eligible_hybrid or is_eligible_baseline:
            result = {
                "Scholarship": scholarship_name,
                "Level": scholarship_level,
                "Hybrid_Model": {
                    "Success_Probability": f"{success_prob_hybrid[i]:.2f}%",
                    "Raw_Score": float(success_prob_hybrid[i]),
                    "Eligibility": "Eligible" if is_eligible_hybrid else "Ineligible"
                },
                "Baseline_Model": {
                    "Success_Probability": f"{success_prob_baseline[i]:.2f}%",
                    "Raw_Score": float(success_prob_baseline[i]),
                    "Eligibility": "Eligible" if is_eligible_baseline else "Ineligible"
                },
                "Average_Probability": f"{((success_prob_hybrid[i] + success_prob_baseline[i]) / 2):.2f}%",
                "Recommendation": "Highly Recommended" if (success_prob_hybrid[i] >= 70 or success_prob_baseline[i] >= 70) else "Recommended"
            }
            
            if disqualified:
                result["Note"] = f"Disqualified: {', '.join(disqualification_reasons)}"
            
            final_results.append(result)
    
    # Sort by average probability
    final_results = sorted(
        final_results,
        key=lambda x: (x['Hybrid_Model']['Raw_Score'] + x['Baseline_Model']['Raw_Score']) / 2,
        reverse=True
    )
    
    return final_results


# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example student profile
    new_student = {
        #Payload utk predict outcome
        'Age': 19,
        'Race': 'Malay',
        'Household Income': 90000,
        'SPM Result (As)': 10,
        'Co-curricular Score': 100,
        'Field of Study': 'Engineering',
        'STPM CGPA': 4.0,
        'Matriculation CGPA': 0.0,
        'Foundation CGPA': 0.0,
        'A-Level (As)': 0,
        'UG CGPA': 0.0
    }
    
    print("="*60)
    print("SCHOLARSHIP MATCHING SYSTEM - PREDICTION")
    print("="*60)
    print("\nStudent Profile:")
    for key, value in new_student.items():
        print(f"  {key}: {value}")
    print("\n")
    
    # Use both models for comparison
    results = predict_with_both_models(new_student)
    print(f"\n✓ Found {len(results)} eligible scholarships\n")
    print(json.dumps(results, indent=2))
    
    # Save results
    with open('new_student_predictions.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Full results saved to 'new_student_predictions.json'")
