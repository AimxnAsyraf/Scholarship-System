"""
Quick test to verify models and data load correctly
"""
import os
import sys
import joblib
import pandas as pd

# Add model_training directory to path
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_training')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

print("="*60)
print("TESTING MODEL AND DATA LOADING")
print("="*60)

# Test 1: Check paths
print(f"\n1. Checking paths:")
print(f"   MODEL_DIR: {MODEL_DIR}")
print(f"   DATASET_DIR: {DATASET_DIR}")
print(f"   MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")
print(f"   DATASET_DIR exists: {os.path.exists(DATASET_DIR)}")

# Test 2: Check model files
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_model.pkl')
print(f"\n2. Checking model files:")
print(f"   Hybrid model exists: {os.path.exists(HYBRID_MODEL_PATH)}")
print(f"   Baseline model exists: {os.path.exists(BASELINE_MODEL_PATH)}")

# Test 3: Load models
try:
    print(f"\n3. Loading models...")
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    print(f"   ✅ Hybrid model loaded: {type(hybrid_model)}")
    print(f"   ✅ Baseline model loaded: {type(baseline_model)}")
except Exception as e:
    print(f"   ❌ Error loading models: {e}")

# Test 4: Load scholarship data
SCHOLARSHIP_CSV_PATH = os.path.join(DATASET_DIR, 'SCHOLARSHIPS.csv')
print(f"\n4. Checking scholarship data:")
print(f"   CSV exists: {os.path.exists(SCHOLARSHIP_CSV_PATH)}")

try:
    df_scholarships = pd.read_csv(SCHOLARSHIP_CSV_PATH)
    df_scholarships.rename(columns={'UG/PG': 'Scholarship Level'}, inplace=True)
    print(f"   ✅ Loaded {len(df_scholarships)} scholarship records")
    
    unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
    print(f"   ✅ Found {len(unique_scholarships)} unique scholarships")
    print(f"\n   First 5 scholarships:")
    for i, row in unique_scholarships.head().iterrows():
        print(f"      - {row['Scholarship Name']} ({row['Scholarship Level']})")
except Exception as e:
    print(f"   ❌ Error loading scholarship data: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
