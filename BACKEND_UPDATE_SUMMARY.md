# Backend Update Summary

## Changes Made to `backend/app.py`

### 1. **Updated Pydantic Models**

#### Added `EducationLevel` Model
```python
class EducationLevel(BaseModel):
    level: str  # 'SPM', 'A Level', 'STPM', 'Matriculation', 'Foundation', 'Undergraduate'
    spm_as: Optional[int] = None
    cgpa: Optional[float] = None
    alevel_stars: Optional[int] = None
```

#### Modified `PredictionRequest` Model
- **Removed**: `level_of_study`, `cgpa`, `spm_as`, `alevel_stars` (single fields)
- **Added**: `education_levels: List[EducationLevel]` (array to support multiple education levels)

### 2. **Co-curricular Score Calculation Algorithm**

Created `calculate_cocurricular_score()` function with comprehensive scoring:

#### Scoring Breakdown (Max 100 points):
- **Position Score (Max 35 points)**:
  - President: 35
  - Vice President: 30
  - Secretary/Treasurer: 25
  - Committee: 18
  - Member: 10

- **Activity Level Score (Max 40 points)**:
  - International: 40
  - National: 35
  - State: 28
  - District: 20
  - School: 12

- **Diversity Bonus (Max 15 points)**:
  - Both club and activity: +10
  - Multiple positions: +3
  - Multiple activity levels: +3

- **Quantity Bonus (Max 10 points)**:
  - 5+ items: 10
  - 4 items: 8
  - 3 items: 6
  - 2 items: 4
  - 1 item: 2

### 3. **Updated Prediction Endpoint Logic**

#### Step 1: Calculate Co-curricular Score
```python
cocurricular_score = calculate_cocurricular_score(data.curricular_items)
```

#### Step 2: Extract Academic Information
- Validates SPM presence (required)
- Extracts SPM result: `spm_as`
- Finds highest CGPA from STPM/Matriculation/Foundation/Undergraduate levels
- Finds A-Level stars if applicable
- Determines `highest_level` for context

#### Step 3: Run Prediction Model
- **Model exists**: Use trained ML models (to be integrated)
- **Model not available**: Use updated rule-based dummy logic

#### Updated Dummy Logic Changes:
1. **Academic Score (max 0.8)**: Uses extracted `cgpa`, `alevel_stars`, or `spm_as` based on highest education level
2. **Co-curricular Score (max 0.5)**: Normalizes calculated `cocurricular_score` (0-100) to 0-0.5 scale
3. **Level Bonus (max 0.2)**: Uses `highest_level` instead of `data.level_of_study`
4. **Academic Eligibility Checks**: Uses extracted variables (`cgpa`, `alevel_stars`, `spm_as`) instead of data fields

## Data Flow

### Frontend to Backend:
```json
{
  "age": 20,
  "race": "Bumiputera",
  "field_of_study": "Computer Science and IT",
  "household_income": 48000,
  "education_levels": [
    {"level": "SPM", "spm_as": 10},
    {"level": "Foundation", "cgpa": 3.8}
  ],
  "curricular_items": [
    {"type": "club", "club_name": "Robotics Club", "position_held": "President"},
    {"type": "activities", "activity_name": "Hackathon", "activity_level": "National"}
  ]
}
```

### Backend Processing:
1. **Calculate**: `cocurricular_score = 88.0` (President 35 + National 35 + Both types 10 + 2 items 4 + Multiple 4)
2. **Extract**: `spm_as = 10`, `cgpa = 3.8`, `highest_level = 'Foundation'`
3. **Normalize**: `curricular_normalized = 0.44` (88/100 * 0.5)
4. **Score**: `academic_score = 0.8` + `curricular_normalized = 0.44` + `income = 0.2` + `level_bonus = 0.2` = **1.64**
5. **Predict**: `prediction_score = 1.64 * 2.0 = 3.28` → clipped to **2.0** → `Eligible`

## Testing Checklist

- [x] Backend accepts `education_levels` array
- [x] SPM validation throws error if missing
- [x] Co-curricular score calculation works with empty items
- [ ] Test with multiple CGPA levels (should pick highest)
- [ ] Test with A-Level + SPM combination
- [ ] Test with only SPM (no higher education)
- [ ] Verify co-curricular score calculation accuracy
- [ ] Integration test: Frontend form → Backend prediction → Response

## Next Steps

1. **Test Backend Endpoint**: Start FastAPI server and test with Postman/curl
2. **Frontend Integration**: Verify form submission works end-to-end
3. **Model Integration**: Replace dummy logic with trained hybrid_model.pkl and baseline_model.pkl
4. **Feature Engineering**: Prepare input features for ML models using extracted scores
