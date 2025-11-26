from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="Scholarship System API",
    description="API for managing scholarship applications and predictions",
    version="1.0.0"
)

# Load ML Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scholarship_pipeline.joblib')
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

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

class PredictionRequest(BaseModel):
    age: int
    race: str
    level_of_study: str
    field_of_study: str
    household_income: float
    curricular_items: List[CurricularItem]
    cgpa: Optional[float] = None
    spm_as: Optional[int] = None
    alevel_stars: Optional[int] = None

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
        # If model exists, use it; otherwise use dummy logic
        if model is not None:
            # Prepare input data as DataFrame (matching training format)
            input_data = pd.DataFrame({
                'CGPA': [data.cgpa],
                'Field of Study': [data.field_of_study],
                'Family Income per Month (RM)': [data.household_income],
                'Co-curricular Score (/100)': [data.extracurricular_score],
                'Number of Leadership Positions': [data.leadership_positions]
            })
            
            # Make prediction
            prediction_score = model.predict(input_data)[0]
            
            # Convert to class (0: Ineligible, 1: In Review, 2: Eligible)
            prediction_class = int(np.round(prediction_score).clip(0, 2))
        else:
            # DUMMY PREDICTION LOGIC (Rule-based)
            print("⚠️ Using dummy prediction logic (model not available)")
            
            # Calculate a simple score based on input features
            score = 0
            
            # Academic performance contribution (max 0.8 points)
            # Convert different academic measures to normalized score
            academic_score = 0
            
            if data.level_of_study == 'SPM':
                # SPM: Number of A's (normalized)
                if data.spm_as is not None:
                    if data.spm_as >= 10:
                        academic_score = 0.8
                    elif data.spm_as >= 8:
                        academic_score = 0.65
                    elif data.spm_as >= 6:
                        academic_score = 0.5
                    elif data.spm_as >= 4:
                        academic_score = 0.3
                    else:
                        academic_score = 0.1
            elif data.level_of_study == 'A Level':
                # A Level: Number of A*
                if data.alevel_stars is not None:
                    if data.alevel_stars >= 4:
                        academic_score = 0.8
                    elif data.alevel_stars >= 3:
                        academic_score = 0.65
                    elif data.alevel_stars >= 2:
                        academic_score = 0.5
                    elif data.alevel_stars >= 1:
                        academic_score = 0.3
                    else:
                        academic_score = 0.1
            else:
                # STPM, Matriculation, Foundation, Undergraduate: CGPA
                if data.cgpa is not None:
                    if data.cgpa >= 3.7:
                        academic_score = 0.8
                    elif data.cgpa >= 3.5:
                        academic_score = 0.65
                    elif data.cgpa >= 3.0:
                        academic_score = 0.5
                    elif data.cgpa >= 2.5:
                        academic_score = 0.3
                    else:
                        academic_score = 0.1
            
            score += academic_score
            
            # Curricular contribution (max 0.5 points) - calculate from all items
            curricular_score = 0
            for item in data.curricular_items:
                item_score = 0
                if item.type == 'club':
                    # Leadership positions score
                    if item.position_held:
                        position_lower = item.position_held.lower()
                        if any(word in position_lower for word in ['president', 'chairman', 'head']):
                            item_score = 0.5
                        elif any(word in position_lower for word in ['vice', 'secretary', 'treasurer']):
                            item_score = 0.35
                        elif any(word in position_lower for word in ['committee', 'member', 'exco']):
                            item_score = 0.2
                        else:
                            item_score = 0.15
                elif item.type == 'activities':
                    # Activity level score
                    level_scores = {
                        'International': 0.5,
                        'National': 0.4,
                        'State': 0.3,
                        'District': 0.2,
                        'School': 0.1
                    }
                    item_score = level_scores.get(item.activity_level, 0.05)
                
                curricular_score = max(curricular_score, item_score)  # Take the highest score
            
            score += curricular_score
            
            # Income consideration (max 0.3 points - lower income = higher score)
            yearly_income = data.household_income
            if yearly_income <= 36000:  # ~RM 3,000/month
                score += 0.3
            elif yearly_income <= 60000:  # ~RM 5,000/month
                score += 0.2
            elif yearly_income <= 96000:  # ~RM 8,000/month
                score += 0.1
            
            # Level of study bonus (max 0.2 points)
            if data.level_of_study in ['Undergraduate', 'Foundation']:
                score += 0.2
            elif data.level_of_study in ['A Level', 'STPM', 'Matriculation']:
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
        if data.level_of_study == 'SPM' and data.spm_as and data.spm_as >= 8:
            has_excellent_academics = True
        elif data.level_of_study == 'A Level' and data.alevel_stars and data.alevel_stars >= 3:
            has_excellent_academics = True
        elif data.cgpa and data.cgpa >= 3.5:
            has_excellent_academics = True
        
        has_good_academics = False
        if data.level_of_study == 'SPM' and data.spm_as and data.spm_as >= 6:
            has_good_academics = True
        elif data.level_of_study == 'A Level' and data.alevel_stars and data.alevel_stars >= 2:
            has_good_academics = True
        elif data.cgpa and data.cgpa >= 3.3:
            has_good_academics = True
        
        # Define scholarships with eligibility criteria
        all_scholarships = [
            {
                "name": "Merit Scholarship",
                "is_eligible": has_excellent_academics,
                "description": "For students with excellent academic performance",
                "weight": 1.2 if has_excellent_academics else 0.6
            },
            {
                "name": "Academic Excellence Award",
                "is_eligible": has_good_academics and (has_leadership or has_high_activity),
                "description": "For high achievers with strong co-curricular involvement",
                "weight": 1.1 if has_good_academics and (has_leadership or has_high_activity) else 0.5
            },
            {
                "name": "Leadership Grant",
                "is_eligible": has_leadership,
                "description": "For students with demonstrated leadership experience",
                "weight": 1.3 if has_leadership else 0.4
            },
            {
                "name": "Need-Based Scholarship",
                "is_eligible": data.household_income <= 60000,  # Yearly income
                "description": "For students with financial need (Yearly Income ≤ RM 60,000)",
                "weight": 1.2 if data.household_income <= 60000 else 0.6
            },
            {
                "name": "Community Service Award",
                "is_eligible": has_community_activity,
                "description": "For students active in community service and co-curricular activities",
                "weight": 1.1 if has_community_activity else 0.5
            },
            {
                "name": "STEM Excellence Scholarship",
                "is_eligible": data.field_of_study in ["Engineering", "Computer Science and IT", "Science", "Medicine and Healthcare"] and has_good_academics,
                "description": "For outstanding students in STEM fields",
                "weight": 1.3 if data.field_of_study in ["Engineering", "Computer Science and IT", "Science", "Medicine and Healthcare"] and has_good_academics else 0.3
            },
            {
                "name": "First Generation Scholarship",
                "is_eligible": data.household_income <= 84000,  # Yearly income
                "description": "Supporting first-generation university students (Yearly Income ≤ RM 84,000)",
                "weight": 1.0 if data.household_income <= 84000 else 0.5
            },
            {
                "name": "Sports & Arts Scholarship",
                "is_eligible": has_high_activity and academic_score >= 0.5,
                "description": "For students excelling in sports or arts with good academic standing",
                "weight": 1.2 if has_high_activity and academic_score >= 0.5 else 0.4
            }
        ]
        
        # Calculate probability for each scholarship
        scholarships_with_probability = []
        for scholarship in all_scholarships:
            # Calculate weighted probability
            probability = base_probability * scholarship["weight"]
            # Adjust by eligibility status
            if prediction_class == 2:  # Eligible
                probability = min(probability * 1.1, 0.98)
            elif prediction_class == 0:  # Ineligible
                probability = probability * 0.3
            
            # Ensure probability is between 0 and 1
            probability = max(0.05, min(0.98, probability))
            
            scholarships_with_probability.append({
                "name": scholarship["name"],
                "probability": round(float(probability), 4),
                "eligibility_status": "Eligible" if scholarship["is_eligible"] else "Not Eligible",
                "description": scholarship["description"]
            })
        
        # Sort by probability (highest first)
        scholarships_with_probability.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "eligible": is_eligible,
            "confidence": float(confidence),
            "prediction_score": float(prediction_score),
            "eligibility_status": status,
            "scholarships": scholarships_with_probability
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
