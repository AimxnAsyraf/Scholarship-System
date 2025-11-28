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
import traceback
import json
from datetime import datetime

# FastAPI static files for serving frontend assets
from fastapi.staticfiles import StaticFiles

# Add model_training directory to path for imports
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_training')
sys.path.insert(0, MODEL_DIR)

# Add dataset directory for loading scholarships
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

# Add output directory for saving predictions
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prediction_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✅ Created output directory: {OUTPUT_DIR}")

app = FastAPI(
    title="Scholarship System API",
    description="API for managing scholarship applications and predictions",
    version="1.0.0"
)

# Mount static files for frontend (css, js, assets)
import pathlib
FRONTEND_STATIC_DIR = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')
if os.path.exists(FRONTEND_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_STATIC_DIR), name="static")
else:
    print(f"⚠️ Static directory not found at {FRONTEND_STATIC_DIR}")

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
SCHOLARSHIP_CSV_PATH = os.path.join(DATASET_DIR, 'SCHOLARSHIPS_2.csv')
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


def save_prediction_to_file(data: PredictionRequest, result: dict) -> str:
    """
    Save prediction results to a JSON file in the output directory.
    Returns the file path where the prediction was saved.
    """
    try:
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with student info
        filename = f"prediction_{timestamp}.json"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Prepare data to save
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "student_info": {
                "age": data.age,
                "race": data.race,
                "field_of_study": data.field_of_study,
                "household_income": data.household_income,
                "education_levels": [
                    {
                        "level": edu.level,
                        "spm_as": edu.spm_as,
                        "cgpa": edu.cgpa,
                        "alevel_stars": edu.alevel_stars
                    }
                    for edu in data.education_levels
                ],
                "curricular_items": [
                    {
                        "type": item.type,
                        "club_name": item.club_name,
                        "position_held": item.position_held,
                        "activity_name": item.activity_name,
                        "activity_level": item.activity_level
                    }
                    for item in data.curricular_items
                ]
            },
            "prediction_results": {
                "eligible": result.get("eligible"),
                "confidence": result.get("confidence"),
                "prediction_score": result.get("prediction_score"),
                "eligibility_status": result.get("eligibility_status"),
                "scholarships": result.get("scholarships", [])
            }
        }
        
        # Write to JSON file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Prediction saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"⚠️ Error saving prediction to file: {e}")
        return None


def calculate_cocurricular_score(curricular_items: List[CurricularItem]) -> float:
    """
    Calculates co-curricular score (0-100) based on clubs and activities.
    
    Scoring Breakdown:
    - Club Leadership Position: Max 35 points
    - Activity Participation Level: Max 40 points  
    - Diversity Bonus: Max 15 points
    - Quantity Bonus: Max 10 points
    """
    
    if not curricular_items or len(curricular_items) == 0:
        return 0.0
    
    score = 0.0
    highest_position_score = 0
    highest_activity_score = 0
    has_club = False
    has_activity = False
    unique_positions = set()
    unique_levels = set()
    
    # Position scoring map (Max 35)
    position_map = {
        'President': 30,
        'Vice President': 25,
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
                position_score = position_map.get(item.position_held, 5)    
                score += position_score
        elif item.type == 'activities':
            has_activity = True
            if item.activity_level:
                activity_score = level_map.get(item.activity_level, 5)
                score += activity_score     

    if score > 100:
        final_score = 100.0
    else:
        final_score = score               
    
    return final_score


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
        # Dynamically import predict_with_both_models from predict_student.py
        import importlib.util
        predict_student_path = os.path.join(MODEL_DIR, "predict_student.py")
        spec = importlib.util.spec_from_file_location("predict_student", predict_student_path)
        predict_student = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_student)
        predict_with_both_models = predict_student.predict_with_both_models

        # Prepare student_profile dict as in predict_student.py
        # Extract academic info
        spm_data = next((edu for edu in data.education_levels if edu.level == 'SPM'), None)
        spm_as = spm_data.spm_as if spm_data and spm_data.spm_as is not None else 0
        alevel_data = next((edu for edu in data.education_levels if edu.level == 'A Level'), None)
        alevel_stars = alevel_data.alevel_stars if alevel_data and alevel_data.alevel_stars is not None else 0
        stpm_cgpa = next((edu.cgpa for edu in data.education_levels if edu.level == 'STPM' and edu.cgpa is not None), 0.0)
        matric_cgpa = next((edu.cgpa for edu in data.education_levels if edu.level == 'Matriculation' and edu.cgpa is not None), 0.0)
        foundation_cgpa = next((edu.cgpa for edu in data.education_levels if edu.level == 'Foundation' and edu.cgpa is not None), 0.0)
        ug_cgpa = next((edu.cgpa for edu in data.education_levels if edu.level == 'Undergraduate' and edu.cgpa is not None), 0.0)

        # Calculate co-curricular score
        cocurricular_score = calculate_cocurricular_score(data.curricular_items)

        student_profile = {
            'Age': data.age,
            'Race': data.race,
            'Household Income': data.household_income,
            'SPM Result (As)': spm_as,
            'Co-curricular Score': cocurricular_score,
            'Field of Study': data.field_of_study,
            'STPM CGPA': stpm_cgpa,
            'Matriculation CGPA': matric_cgpa,
            'Foundation CGPA': foundation_cgpa,
            'A-Level (As)': alevel_stars,
            'UG CGPA': ug_cgpa
        }

        # Call predict_with_both_models
        results = predict_with_both_models(student_profile)

        # Prepare API response (flatten for API) - Filter by average probability > 30%
        scholarships = []
        for res in results:
            # Parse probabilities from strings
            avg_prob_str = res["Average_Probability"].replace('%', '')
            avg_prob = float(avg_prob_str)
            
            hybrid_prob_str = res["Hybrid_Model"]["Success_Probability"].replace('%', '')
            hybrid_prob = float(hybrid_prob_str)
            
            baseline_prob_str = res["Baseline_Model"]["Success_Probability"].replace('%', '')
            baseline_prob = float(baseline_prob_str)
            
            # Only include scholarships with average probability >= 30%
            if avg_prob >= 30:
                chosen_prob = hybrid_prob
                chosen_model = "Hybrid"
                # Determine eligibility status from both models
                hybrid_eligible = res["Hybrid_Model"]["Eligibility"] == "Eligible"
                baseline_eligible = res["Baseline_Model"]["Eligibility"] == "Eligible"
                
                # # Apply probability selection rules
                # if hybrid_eligible and baseline_eligible:
                #     # Both models mark as eligible - choose higher probability
                #     chosen_prob = max(hybrid_prob, baseline_prob)
                #     chosen_model = "Hybrid" if hybrid_prob >= baseline_prob else "Baseline"
                # elif hybrid_eligible:
                #     # Only Hybrid marks as eligible
                #     chosen_prob = hybrid_prob
                #     chosen_model = "Hybrid"
                # elif baseline_eligible:
                #     # Only Baseline marks as eligible
                #     chosen_prob = baseline_prob
                #     chosen_model = "Baseline"
                # else:
                #     # Neither marks as eligible but average >= 30%
                #     # Use higher probability between the two
                #     chosen_prob = max(hybrid_prob, baseline_prob)
                #     chosen_model = "Hybrid" if hybrid_prob >= baseline_prob else "Baseline"
                
                scholarships.append({
                    "name": res["Scholarship"],
                    "probability": chosen_prob / 100.0,  # normalized
                    "eligibility_status": "Eligible" if (hybrid_eligible or baseline_eligible) else "Ineligible",
                    "description": f"{res['Level']} level scholarship",
                    "hybrid_probability": res["Hybrid_Model"]["Success_Probability"],
                    "hybrid_probability_value": hybrid_prob,
                    "baseline_probability": res["Baseline_Model"]["Success_Probability"],
                    "baseline_probability_value": baseline_prob,
                    "average_probability": res["Average_Probability"],
                    "average_probability_value": avg_prob,
                    "hybrid_eligibility": res["Hybrid_Model"]["Eligibility"],
                    "baseline_eligibility": res["Baseline_Model"]["Eligibility"],
                    "chosen_probability": chosen_prob,
                    "chosen_model": chosen_model,
                    "recommendation": res["Recommendation"]
                })

        # Compute summary stats for API
        eligible = len(scholarships) > 0
        prediction_score = max([s["probability"] for s in scholarships], default=0)
        confidence = 0.99 if eligible else 0.5
        eligibility_status = "Eligible" if eligible else "Ineligible"

        response = {
            "eligible": eligible,
            "confidence": confidence,
            "prediction_score": prediction_score,
            "eligibility_status": eligibility_status,
            "scholarships": scholarships
        }

        # Save prediction to JSON file
        saved_file = save_prediction_to_file(data, response)
        if saved_file:
            print(f"📁 Prediction output saved successfully")

        return response
    except Exception as e:
        print(f"❌ Error in predict_eligibility: {e}")
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
