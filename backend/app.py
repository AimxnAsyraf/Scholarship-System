from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Scholarship System API",
    description="API for managing scholarship applications and predictions",
    version="1.0.0"
)

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

class PredictionRequest(BaseModel):
    gpa: float
    income: float
    extracurricular_score: int

class PredictionResponse(BaseModel):
    eligible: bool
    confidence: float
    recommended_scholarships: List[str]

# Routes
@app.get("/")
def home():
    return {
        "message": "Welcome to Scholarship System API",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/api/scholarships", response_model=List[Scholarship])
def get_scholarships():
    """Get all available scholarships"""
    # TODO: Implement database query
    scholarships = [
        {
            "id": 1,
            "name": "Merit Scholarship",
            "amount": 5000.0,
            "description": "For students with excellent academic performance"
        },
        {
            "id": 2,
            "name": "Need-Based Scholarship",
            "amount": 3000.0,
            "description": "For students with financial need"
        }
    ]
    return scholarships

@app.get("/api/scholarships/{scholarship_id}", response_model=Scholarship)
def get_scholarship(scholarship_id: int):
    """Get a specific scholarship by ID"""
    # TODO: Implement database query
    scholarship = {
        "id": scholarship_id,
        "name": "Merit Scholarship",
        "amount": 5000.0,
        "description": "For students with excellent academic performance"
    }
    return scholarship

# @app.post("/api/applications", response_model=ApplicationResponse, status_code=201)
# def submit_application(application: Application):
#     """Submit a new scholarship application"""
#     # TODO: Validate and save application to database
#     return {
#         "id": 123,
#         "status": "pending",
#         "submitted_date": "2025-11-24"
#     }

# @app.get("/api/applications/{application_id}", response_model=ApplicationResponse)
# def get_application(application_id: int):
#     """Get application status by ID"""
#     # TODO: Implement database query
#     application = {
#         "id": application_id,
#         "status": "pending",
#         "submitted_date": "2025-11-24"
#     }
#     return application

@app.post("/api/predict", response_model=PredictionResponse)
def predict_eligibility(data: PredictionRequest):
    """Predict scholarship eligibility using ML model"""
    # TODO: Load and use ML model for prediction
    prediction = {
        "eligible": True,
        "confidence": 0.85,
        "recommended_scholarships": ["Merit Scholarship", "Academic Excellence Award"]
    }
    return prediction

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
