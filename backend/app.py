from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add model_training directory to path for imports
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_training')
sys.path.insert(0, MODEL_DIR)

# Add dataset directory for loading scholarships
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

app = FastAPI(
    title="Scholarship System API",
    description="API for managing scholarship applications and predictions",
    version="1.0.0"
)

# Load ML Models (Both Hybrid and Baseline)
HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model.pkl')
BASELINE_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_model.pkl')

hybrid_model = None
baseline_model = None

try:
    if os.path.exists(HYBRID_MODEL_PATH):
        hybrid_model = joblib.load(HYBRID_MODEL_PATH)
        print(f"✅ Hybrid model loaded from {HYBRID_MODEL_PATH}")
    else:
        print(f"⚠️ Hybrid model not found at {HYBRID_MODEL_PATH}")
        
    if os.path.exists(BASELINE_MODEL_PATH):
        baseline_model = joblib.load(BASELINE_MODEL_PATH)
        print(f"✅ Baseline model loaded from {BASELINE_MODEL_PATH}")
    else:
        print(f"⚠️ Baseline model not found at {BASELINE_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Load Scholarship Data from CSV
SCHOLARSHIP_CSV_PATH = os.path.join(DATASET_DIR, 'SCHOLARSHIPS.csv')
df_scholarships = None

try:
    if os.path.exists(SCHOLARSHIP_CSV_PATH):
        df_scholarships = pd.read_csv(SCHOLARSHIP_CSV_PATH)
        # Rename column for consistency
        df_scholarships.rename(columns={'UG/PG': 'Scholarship Level'}, inplace=True)
        print(f"✅ Loaded {len(df_scholarships)} scholarship records from {SCHOLARSHIP_CSV_PATH}")
    else:
        print(f"⚠️ Scholarship data not found at {SCHOLARSHIP_CSV_PATH}")
except Exception as e:
    print(f"❌ Error loading scholarship data: {e}")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class Scholarship(BaseModel):
    id: int
    name: str
    amount: float
    description: str

class Application(BaseModel):
    student_name: str
    email: str
    gpa: float
    scholarship_id: int

class ApplicationResponse(BaseModel):
    id: int
    status: str
    submitted_date: str

class CurricularItem(BaseModel):
    type: str
    club_name: Optional[str] = None
    position_held: Optional[str] = None
    activity_name: Optional[str] = None
    activity_level: Optional[str] = None

class EducationLevel(BaseModel):
    level: str
    spm_as: Optional[int] = None
    cgpa: Optional[float] = None
    alevel_stars: Optional[int] = None

class PredictionRequest(BaseModel):
    age: int
    race: str
    field_of_study: str
    household_income: float
    education_levels: List[EducationLevel]
    curricular_items: List[CurricularItem]

class ScholarshipProbability(BaseModel):
    name: str
    probability: float
    eligibility_status: str
    description: str

class PredictionResponse(BaseModel):
    eligible: bool
    confidence: float
    prediction_score: float
    eligibility_status: str
    scholarships: List[ScholarshipProbability]

# ==========================================
# CO-CURRICULAR SCORE CALCULATION ALGORITHM
# ==========================================
def calculate_cocurricular_score(curricular_items: List[CurricularItem]) -> float:
    """
    Calculates co-curricular score (0-100) based on clubs and activities.
    
    Scoring Breakdown:
    - Club Leadership Position: Max 35 points
    - Activity Participation Level: Max 40 points  
    - Diversity Bonus: Max 15 points
    - Quantity Bonus: Max 10 points
    
    Args:
        curricular_items: List of clubs and activities from form
    
    Returns:
        float: Score between 0-100
    """
    
    if not curricular_items or len(curricular_items) == 0:
        return 0.0
    
    score = 0.0
    
    # Track highest scores for each category
    highest_position_score = 0
    highest_activity_score = 0
    
    # Track diversity
    has_club = False
    has_activity = False
    unique_positions = set()
    unique_levels = set()
    
    # Position scoring map (Max 35)
    position_map = {
        'President': 30,
        'Vice President': 25 ,
        'Secretary': 20,
        'Treasurer': 20,
        'Committee': 15,
        'Member': 10
    }
    
    # Activity level scoring map (Max 40)
    level_map = {
        'International': 30,
        'National': 25,
        'State': 20,
        'District': 15,
        'School': 10
    }
    
    # Process each curricular item
    for item in curricular_items:
        if item.type == 'club':
            has_club = True
            if item.position_held:
                unique_positions.add(item.position_held)
                position_score = position_map.get(item.position_held, 5)
                highest_position_score = max(highest_position_score, position_score)
        
        elif item.type == 'activities':
            has_activity = True
            if item.activity_level:
                unique_levels.add(item.activity_level)
                activity_score = level_map.get(item.activity_level, 5)
                highest_activity_score = max(highest_activity_score, activity_score)
    
    # Add highest scores from each category
    score += highest_position_score
    score += highest_activity_score
    
    # Diversity Bonus (Max 15 points)
    diversity_bonus = 0
    if has_club and has_activity:
        diversity_bonus += 10  # Bonus for having both types
    
    # Additional diversity for multiple positions/levels
    if len(unique_positions) >= 2:
        diversity_bonus += 3
    if len(unique_levels) >= 2:
        diversity_bonus += 3
    
    diversity_bonus = min(diversity_bonus, 15)
    score += diversity_bonus
    
    # Quantity Bonus (Max 10 points)
    # Reward students with multiple involvements
    num_items = len(curricular_items)
    if num_items >= 5:
        quantity_bonus = 10
    elif num_items >= 4:
        quantity_bonus = 8
    elif num_items >= 3:
        quantity_bonus = 6
    elif num_items >= 2:
        quantity_bonus = 4
    else:
        quantity_bonus = 2
    
    score += quantity_bonus
    
    # Cap at 100 and round to 2 decimal places
    final_score = min(100.0, round(score, 2))
    
    return final_score


# Routes
@app.get("/")
def home():
    return {
        "message": "Welcome to Scholarship System API",
        "status": "running",
        "docs": "/docs"
    }

@app.post("/api/predict", response_model=PredictionResponse)
def predict_eligibility(data: PredictionRequest):
    """Predict scholarship eligibility using ML model (or dummy logic if model not available)"""
    
    try:
        # ==========================================
        # STEP 1: Calculate Co-curricular Score
        # ==========================================
        cocurricular_score = calculate_cocurricular_score(data.curricular_items)
        print(f"📊 Calculated Co-curricular Score: {cocurricular_score}/100")
        
        # ==========================================
        # STEP 2: Extract Academic Information
        # ==========================================
        # Find SPM result (required)
        spm_data = next((edu for edu in data.education_levels if edu.level == 'SPM'), None)
        if not spm_data:
            raise HTTPException(status_code=400, detail="SPM result is required")
        
        spm_as = spm_data.spm_as
        
        # Find highest education level for CGPA/A-Level
        cgpa = None
        alevel_stars = None
        highest_level = 'SPM'
        
        for edu in data.education_levels:
            if edu.level == 'A Level' and edu.alevel_stars is not None:
                alevel_stars = edu.alevel_stars
                highest_level = 'A Level'
            elif edu.level in ['STPM', 'Matriculation', 'Foundation', 'Undergraduate']:
                if edu.cgpa is not None:
                    # Keep the highest CGPA or the most recent one
                    if cgpa is None or edu.cgpa > cgpa:
                        cgpa = edu.cgpa
                        highest_level = edu.level
        
        print(f"📚 Academic Data - SPM: {spm_as} A's, Highest Level: {highest_level}, CGPA: {cgpa}, A-Level: {alevel_stars}")
        
        # ==========================================
        # STEP 3: Run Prediction Model
        # ==========================================
        # Use both trained models if available
        model_predictions_available = False
        hybrid_predictions = None
        baseline_predictions = None
        unique_scholarships = None
        
        if hybrid_model is not None and baseline_model is not None:
            print("🤖 Using trained ML models for prediction...")
            print(f"🔍 Models loaded: Hybrid={hybrid_model is not None}, Baseline={baseline_model is not None}")
            
            try:
                # Get unique scholarships from loaded CSV data
                if df_scholarships is None:
                    raise Exception("Scholarship data not loaded")
                
                unique_scholarships = df_scholarships[['Scholarship Name', 'Scholarship Level']].drop_duplicates()
                num_scholarships = len(unique_scholarships)
                print(f"📚 Loaded {num_scholarships} unique scholarships from dataset")
                
                # Prepare input features
                student_data = {
                    'Age': data.age,
                    'Race': data.race,
                    'Household Income': data.household_income,
                    'SPM Result (As)': spm_as if spm_as else 0,
                    'Co-curricular Score': cocurricular_score,
                    'Field of Study': data.field_of_study,
                    'STPM CGPA': cgpa if highest_level == 'STPM' else 0.0,
                    'Matriculation CGPA': cgpa if highest_level == 'Matriculation' else 0.0,
                    'Foundation CGPA': cgpa if highest_level == 'Foundation' else 0.0,
                    'A-Level (As)': alevel_stars if alevel_stars else 0,
                    'UG CGPA': cgpa if highest_level == 'Undergraduate' else 0.0
                }
                
                # Calculate Unified CGPA
                cgpa_sources = ['STPM CGPA', 'Matriculation CGPA', 'Foundation CGPA', 'UG CGPA']
                cgpa_unified = max([student_data.get(src, 0) for src in cgpa_sources])
                student_data['CGPA_Unified'] = cgpa_unified
                
                # Create DataFrame with repeated student data for each scholarship
                test_df = pd.DataFrame([student_data] * num_scholarships)
                
                # Add actual scholarship information
                test_df['Applied Scholarship'] = unique_scholarships['Scholarship Name'].values
                test_df['Scholarship Level'] = unique_scholarships['Scholarship Level'].values
                
                # Merge with scholarship details from CSV
                test_merged = pd.merge(
                    test_df,
                    df_scholarships,
                    left_on=['Applied Scholarship', 'Scholarship Level'],
                    right_on=['Scholarship Name', 'Scholarship Level'],
                    how='left'
                )
                
                # Import feature engineering functions (path added to sys.path at startup)
                from gemini_train import apply_fuzzy_logic, add_engineered_features  # type: ignore
                
                # Apply feature engineering
                test_processed = apply_fuzzy_logic(test_merged)
                test_processed = add_engineered_features(test_processed)
                
                # Predict with both models
                hybrid_predictions = np.clip(hybrid_model.predict(test_processed), 0, 100)
                baseline_predictions = np.clip(baseline_model.predict(test_processed), 0, 100)
                
                # Calculate average predictions for overall score
                avg_predictions = (hybrid_predictions + baseline_predictions) / 2
                
                # Use average of all scholarship predictions as overall prediction_score
                prediction_score = np.mean(avg_predictions) / 50  # Normalize to 0-2 scale
                prediction_class = int(np.round(prediction_score).clip(0, 2))
                
                print(f"📊 Model Predictions - Hybrid Avg: {np.mean(hybrid_predictions):.2f}%, Baseline Avg: {np.mean(baseline_predictions):.2f}%")
                print(f"📊 Overall Average: {np.mean(avg_predictions):.2f}%, Prediction Score: {prediction_score:.2f}")
                print(f"📊 Predictions array length: {len(hybrid_predictions)}")
                
                # Mark that model predictions are available
                model_predictions_available = True
                print("✅ Model predictions available = True")
                
            except Exception as e:
                print(f"⚠️ Error during model prediction: {e}")
                import traceback
                print(f"⚠️ Traceback: {traceback.format_exc()}")
                print("⚠️ Falling back to dummy logic")
                model_predictions_available = False
        
        if not model_predictions_available:
            # DUMMY PREDICTION LOGIC (Rule-based)
            print("⚠️ Using dummy prediction logic (model not available)")
            
            # Calculate a simple score based on input features
            score = 0
            
            # Academic performance contribution (max 0.8 points)
            # Use extracted academic data from education_levels array
            academic_score = 0
            
            # Prioritize highest education level
            if cgpa is not None:
                # CGPA-based (STPM, Matriculation, Foundation, Undergraduate)
                if cgpa >= 3.7:
                    academic_score = 0.8
                elif cgpa >= 3.5:
                    academic_score = 0.65
                elif cgpa >= 3.0:
                    academic_score = 0.5
                elif cgpa >= 2.5:
                    academic_score = 0.3
                else:
                    academic_score = 0.1
            elif alevel_stars is not None:
                # A Level: Number of A*
                if alevel_stars >= 4:
                    academic_score = 0.8
                elif alevel_stars >= 3:
                    academic_score = 0.65
                elif alevel_stars >= 2:
                    academic_score = 0.5
                elif alevel_stars >= 1:
                    academic_score = 0.3
                else:
                    academic_score = 0.1
            elif spm_as is not None:
                # SPM: Number of A's (fallback if no higher education)
                if spm_as >= 10:
                    academic_score = 0.8
                elif spm_as >= 8:
                    academic_score = 0.65
                elif spm_as >= 6:
                    academic_score = 0.5
                elif spm_as >= 4:
                    academic_score = 0.3
                else:
                    academic_score = 0.1
            
            score += academic_score
            
            # Co-curricular contribution (max 0.5 points)
            # Use the already calculated co-curricular score (0-100) and normalize to 0-0.5
            curricular_normalized = (cocurricular_score / 100) * 0.5
            score += curricular_normalized
            print(f"💯 Normalized Co-curricular: {curricular_normalized:.2f}/0.5")
            
            # Income consideration (max 0.3 points - lower income = higher score)
            yearly_income = data.household_income
            if yearly_income <= 36000:  # ~RM 3,000/month
                score += 0.3
            elif yearly_income <= 60000:  # ~RM 5,000/month
                score += 0.2
            elif yearly_income <= 96000:  # ~RM 8,000/month
                score += 0.1
            
            # Level of study bonus (max 0.2 points) - based on highest education level
            if highest_level in ['Undergraduate', 'Foundation']:
                score += 0.2
            elif highest_level in ['A Level', 'STPM', 'Matriculation']:
                score += 0.15
            else:
                score += 0.1
            
            # Convert score to prediction_score (0-2 scale)
            prediction_score = score * 2.0  # Scale from 0-2
            prediction_class = int(np.round(prediction_score).clip(0, 2))
        
        # Map to eligibility status
        eligibility_map = {0: 'Ineligible', 1: 'In Review', 2: 'Eligible'}
        status = eligibility_map[prediction_class]
        
        # Calculate confidence (distance from nearest integer)
        confidence = 1.0 - abs(prediction_score - prediction_class)
        confidence = max(0.5, min(0.99, confidence))  # Ensure reasonable confidence range
        
        # Determine eligibility
        is_eligible = prediction_class >= 1  # In Review or Eligible
        
        # Calculate probability scores for each scholarship
        # Base probability from prediction score (normalized to 0-1)
        base_probability = prediction_score / 2.0  # Since max class is 2
        
        # Check for leadership position from curricular items
        has_leadership = False
        for item in data.curricular_items:
            if item.type == 'club' and item.position_held:
                position_lower = item.position_held.lower()
                if any(word in position_lower for word in ['president', 'chairman', 'head', 'vice', 'secretary', 'treasurer']):
                    has_leadership = True
                    break
        
        # Check for high-level activities from curricular items
        has_high_activity = False
        has_community_activity = False
        for item in data.curricular_items:
            if item.type == 'activities':
                if item.activity_level in ['International', 'National', 'State']:
                    has_high_activity = True
                if item.activity_level in ['State', 'District', 'School']:
                    has_community_activity = True
        
        # Check if student has excellent academic performance (normalized across all levels)
        has_excellent_academics = False
        if cgpa and cgpa >= 3.5:
            has_excellent_academics = True
        elif alevel_stars and alevel_stars >= 3:
            has_excellent_academics = True
        elif spm_as and spm_as >= 8 and highest_level == 'SPM':
            has_excellent_academics = True
        
        has_good_academics = False
        if cgpa and cgpa >= 3.3:
            has_good_academics = True
        elif alevel_stars and alevel_stars >= 2:
            has_good_academics = True
        elif spm_as and spm_as >= 6 and highest_level == 'SPM':
            has_good_academics = True
        
        # ==========================================
        # Define Scholarships with Eligibility Criteria
        # ==========================================
        # Each scholarship has specific requirements that students must meet
        
        all_scholarships = [
            {
                "name": "Merit Scholarship",
                "is_eligible": has_excellent_academics,
                "description": "For students with excellent academic performance (CGPA ≥ 3.5, A-Level ≥ 3A*, or SPM ≥ 8A)",
                "weight": 1.3
            },
            {
                "name": "Academic Excellence Award",
                "is_eligible": has_good_academics and (has_leadership or has_high_activity),
                "description": "For high achievers with strong co-curricular involvement",
                "weight": 1.2
            },
            {
                "name": "Leadership Scholarship",
                "is_eligible": has_leadership and has_good_academics,
                "description": "For students with demonstrated leadership experience and good academic standing",
                "weight": 1.25
            },
            {
                "name": "Need-Based Scholarship",
                "is_eligible": data.household_income <= 60000 and has_good_academics,
                "description": "For financially disadvantaged students with good academic performance (Income ≤ RM 60,000/year)",
                "weight": 1.3
            },
            {
                "name": "Bumiputera Excellence Award",
                "is_eligible": data.race.lower() == "bumiputera" and has_excellent_academics,
                "description": "For outstanding Bumiputera students with excellent academic achievements",
                "weight": 1.2
            },
            {
                "name": "Community Service Award",
                "is_eligible": has_community_activity and cocurricular_score >= 50,
                "description": "For students active in community service with strong co-curricular involvement (Co-curricular Score ≥ 50)",
                "weight": 1.15
            },
            {
                "name": "STEM Excellence Scholarship",
                "is_eligible": data.field_of_study in ["Engineering", "Computer Science and IT", "Science", "Medicine and Healthcare"] and has_excellent_academics,
                "description": "For outstanding students pursuing STEM fields",
                "weight": 1.35
            },
            {
                "name": "First Generation Scholarship",
                "is_eligible": data.household_income <= 84000 and has_good_academics,
                "description": "Supporting first-generation university students (Income ≤ RM 84,000/year)",
                "weight": 1.1
            },
            {
                "name": "Sports & Arts Excellence",
                "is_eligible": has_high_activity and cocurricular_score >= 60 and academic_score >= 0.5,
                "description": "For students excelling in sports or arts with good academic standing (Co-curricular ≥ 60)",
                "weight": 1.25
            },
            {
                "name": "All-Rounder Scholarship",
                "is_eligible": has_excellent_academics and has_leadership and has_high_activity,
                "description": "For well-rounded students excelling in academics, leadership, and activities",
                "weight": 1.4
            },
            {
                "name": "Future Leaders Grant",
                "is_eligible": has_leadership and cocurricular_score >= 70,
                "description": "For emerging leaders with exceptional co-curricular involvement (Co-curricular ≥ 70)",
                "weight": 1.3
            },
            {
                "name": "Undergraduate Achievement Award",
                "is_eligible": highest_level == "Undergraduate" and cgpa and cgpa >= 3.3,
                "description": "For undergraduate students maintaining strong academic performance",
                "weight": 1.15
            },
            {
                "name": "Foundation Excellence Grant",
                "is_eligible": highest_level == "Foundation" and has_excellent_academics,
                "description": "For foundation students with excellent academic achievements",
                "weight": 1.1
            },
            {
                "name": "Low Income Support Fund",
                "is_eligible": data.household_income <= 36000,
                "description": "Critical financial support for students from low-income families (Income ≤ RM 36,000/year)",
                "weight": 1.35
            },
            {
                "name": "Middle Income Scholarship",
                "is_eligible": 36000 < data.household_income <= 96000 and has_good_academics,
                "description": "Supporting middle-income families with qualified students (Income RM 36,000 - RM 96,000/year)",
                "weight": 1.0
            }
        ]
        
        # ==========================================
        # STEP 4: Filter Scholarships Based on Model Results
        # ==========================================
        # LOGIC (following predict_student.py):
        # 1. If average probability > 30%, consider the scholarship
        # 2. Get results from model that produces "Eligible" status (prob >= 40 AND passes hard rules)
        # 3. If both models are "Eligible", choose the one with higher success probability
        eligible_scholarships = []
        
        if model_predictions_available and hybrid_predictions is not None and baseline_predictions is not None:
            print("🎯 Applying model predictions with hard rules from dataset...")
            
            bumiputera_races = ['Malay', 'Bumiputera', 'Bumiputera Sabah / Sarawak']
            
            for i, row in unique_scholarships.iterrows():
                scholarship_name = row['Scholarship Name']
                scholarship_level = row['Scholarship Level']
                
                # Get scholarship rules from CSV
                rule_row = df_scholarships[
                    (df_scholarships['Scholarship Name'] == scholarship_name) &
                    (df_scholarships['Scholarship Level'] == scholarship_level)
                ].iloc[0]
                
                is_bumi_only = rule_row['Bumiputera Only'] == 'Yes'
                offered_fields = str(rule_row['Offered field of study']).split(', ')
                
                # Get predictions for this scholarship
                hybrid_prob = hybrid_predictions[i]
                baseline_prob = baseline_predictions[i]
                avg_prob = (hybrid_prob + baseline_prob) / 2
                
                # Check hard rules (disqualification)
                disqualified = False
                disqualification_reasons = []
                
                # Rule 1: Bumiputera requirement
                if is_bumi_only and (data.race not in bumiputera_races):
                    disqualified = True
                    disqualification_reasons.append("Not Bumiputera")
                
                # Rule 2: PG scholarship requires UG degree
                if scholarship_level == 'PG' and (cgpa is None or highest_level != 'Undergraduate'):
                    disqualified = True
                    disqualification_reasons.append("No UG degree for PG scholarship")
                
                # Determine eligibility for each model (threshold: 40%)
                is_eligible_hybrid = (hybrid_prob >= 40) and (not disqualified)
                is_eligible_baseline = (baseline_prob >= 40) and (not disqualified)
                
                # RULE 1: Average Probability > 30%
                if avg_prob > 30:
                    # RULE 2: At least one model must predict "Eligible"
                    if is_eligible_hybrid or is_eligible_baseline:
                        # RULE 3: Determine which model result to use
                        if is_eligible_hybrid and is_eligible_baseline:
                            # Both eligible: Use higher probability
                            chosen_model = "Hybrid" if hybrid_prob >= baseline_prob else "Baseline"
                            chosen_prob = max(hybrid_prob, baseline_prob)
                        elif is_eligible_hybrid:
                            # Only hybrid eligible
                            chosen_model = "Hybrid"
                            chosen_prob = hybrid_prob
                        else:
                            # Only baseline eligible
                            chosen_model = "Baseline"
                            chosen_prob = baseline_prob
                        
                        # Build description from CSV data
                        description = f"{scholarship_level} level scholarship"
                        if offered_fields and offered_fields[0] != 'nan':
                            description += f" for {', '.join(offered_fields[:3])}"
                        
                        eligible_scholarships.append({
                            "name": scholarship_name,
                            "level": scholarship_level,
                            "probability": round(float(chosen_prob / 100), 4),  # Use chosen model's probability
                            "eligibility_status": "Eligible",
                            "description": description,
                            "hybrid_probability": f"{hybrid_prob:.2f}%",
                            "baseline_probability": f"{baseline_prob:.2f}%",
                            "average_probability": f"{avg_prob:.2f}%",
                            "hybrid_eligibility": "Eligible" if is_eligible_hybrid else "Ineligible",
                            "baseline_eligibility": "Eligible" if is_eligible_baseline else "Ineligible",
                            "chosen_model": chosen_model
                        })
            
            print(f"✅ Found {len(eligible_scholarships)} eligible scholarships (avg > 30%, at least one model eligible)")
            if len(eligible_scholarships) > 0:
                print(f"📋 First scholarship: {eligible_scholarships[0]['name']} - {eligible_scholarships[0]['level']}")
        
        else:
            # Fallback to dummy logic without models
            print(f"📋 Using dummy logic for scholarship matching... (model_predictions_available={model_predictions_available})")
            
            for scholarship in all_scholarships:
                if scholarship["is_eligible"]:
                    # Calculate weighted probability
                    probability = base_probability * scholarship["weight"]
                    
                    # Adjust by prediction class
                    if prediction_class == 2:  # Eligible
                        probability = min(probability * 1.15, 0.95)
                    elif prediction_class == 1:  # In Review
                        probability = min(probability * 1.0, 0.85)
                    else:  # Ineligible
                        probability = probability * 0.5
                    
                    # Ensure probability is between 0.10 and 0.95
                    probability = max(0.10, min(0.95, probability))
                    
                    eligible_scholarships.append({
                        "name": scholarship["name"],
                        "probability": round(float(probability), 4),
                        "eligibility_status": "Eligible",
                        "description": scholarship["description"]
                    })
            
            print(f"✅ Found {len(eligible_scholarships)} eligible scholarships (dummy logic)")
        
        # Sort by probability (highest first)
        eligible_scholarships.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "eligible": is_eligible,
            "confidence": float(confidence),
            "prediction_score": float(prediction_score),
            "eligibility_status": status,
            "scholarships": eligible_scholarships
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
