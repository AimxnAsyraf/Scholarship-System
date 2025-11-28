// Store curricular items and education levels
let curricularItems = [];
let educationLevels = [];
let hasSPM = false; // Track if SPM has been added

// Form Progress Tracking
function updateFormProgress() {
    const form = document.getElementById('predict-form');
    const progressBar = document.getElementById('form-progress');
    const requiredFields = form.querySelectorAll('[required]');
    let filledFields = 0;
    
    requiredFields.forEach(field => {
        if (field.value && field.value.trim() !== '') {
            filledFields++;
        }
    });
    
    // Add curricular items to progress
    if (curricularItems.length > 0) {
        filledFields += 0.5; // Half point for having curricular items
    }
    
    const progress = (filledFields / (requiredFields.length + 0.5)) * 100;
    progressBar.style.width = Math.min(progress, 100) + '%';
}

// Track changes on all form inputs
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predict-form');
    const inputs = form.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('input', updateFormProgress);
        input.addEventListener('change', updateFormProgress);
    });
    
    // Initial progress check
    updateFormProgress();
});

// Handle dynamic academic performance fields based on level of study
document.getElementById('level_of_study').addEventListener('change', function() {
    const level = this.value;
    const spmField = document.getElementById('spm-field');
    const cgpaField = document.getElementById('cgpa-field');
    const alevelField = document.getElementById('alevel-field');
    
    // Hide all fields initially
    spmField.style.display = 'none';
    cgpaField.style.display = 'none';
    alevelField.style.display = 'none';
    
    // Check if trying to add non-SPM level first
    if (level && level !== 'SPM' && !hasSPM) {
        alert('Please add SPM result first before adding other education levels');
        this.value = '';
        return;
    }
    
    // Show relevant field
    if (level === 'SPM') {
        spmField.style.display = 'block';
    } else if (level === 'A Level') {
        alevelField.style.display = 'block';
    } else if (level === 'STPM' || level === 'Matriculation' || level === 'Foundation' || level === 'Undergraduate') {
        cgpaField.style.display = 'block';
    }
});

// Add SPM result
document.getElementById('add-spm-btn').addEventListener('click', function() {
    const spmAs = document.getElementById('spm_as').value.trim();
    
    if (!spmAs || spmAs < 0 || spmAs > 20) {
        alert('Please enter a valid number of A\'s (0-20)');
        return;
    }
    
    // Check if SPM already added
    if (hasSPM) {
        alert('SPM result has already been added');
        return;
    }
    
    educationLevels.push({
        level: 'SPM',
        spm_as: parseInt(spmAs)
    });
    
    hasSPM = true;
    updateEducationList();
    
    // Clear inputs and hide field
    document.getElementById('spm_as').value = '';
    document.getElementById('spm-field').style.display = 'none';
    document.getElementById('level_of_study').value = '';
});

// Add CGPA-based level
document.getElementById('add-cgpa-btn').addEventListener('click', function() {
    const level = document.getElementById('level_of_study').value;
    const cgpa = document.getElementById('cgpa').value.trim();
    
    if (!cgpa || cgpa < 0 || cgpa > 4.0) {
        alert('Please enter a valid CGPA (0.0-4.0)');
        return;
    }
    
    // Check if this level already added
    if (educationLevels.some(item => item.level === level)) {
        alert(`${level} result has already been added`);
        return;
    }
    
    educationLevels.push({
        level: level,
        cgpa: parseFloat(cgpa)
    });
    
    updateEducationList();
    
    // Clear inputs and hide field
    document.getElementById('cgpa').value = '';
    document.getElementById('cgpa-field').style.display = 'none';
    document.getElementById('level_of_study').value = '';
});

// Add A-Level result
document.getElementById('add-alevel-btn').addEventListener('click', function() {
    const alevelStars = document.getElementById('alevel_stars').value.trim();
    
    if (!alevelStars || alevelStars < 0 || alevelStars > 4) {
        alert('Please enter a valid number of A* (0-4)');
        return;
    }
    
    // Check if A-Level already added
    if (educationLevels.some(item => item.level === 'A Level')) {
        alert('A-Level result has already been added');
        return;
    }
    
    educationLevels.push({
        level: 'A Level',
        alevel_stars: parseInt(alevelStars)
    });
    
    updateEducationList();
    
    // Clear inputs and hide field
    document.getElementById('alevel_stars').value = '';
    document.getElementById('alevel-field').style.display = 'none';
    document.getElementById('level_of_study').value = '';
});

// Update the displayed education list
function updateEducationList() {
    const listContainer = document.getElementById('education-list');
    const itemsList = document.getElementById('education-items');
    
    if (educationLevels.length === 0) {
        listContainer.style.display = 'none';
        updateFormProgress();
        return;
    }
    
    listContainer.style.display = 'block';
    itemsList.innerHTML = '';
    
    educationLevels.forEach((item, index) => {
        const li = document.createElement('li');
        let displayText = `<strong>${item.level}:</strong> `;
        
        if (item.level === 'SPM') {
            displayText += `${item.spm_as} A's`;
        } else if (item.level === 'A Level') {
            displayText += `${item.alevel_stars} A*`;
        } else {
            displayText += `CGPA ${item.cgpa}`;
        }
        
        // SPM cannot be removed if other education levels exist
        const isSPMWithOthers = (item.level === 'SPM' && educationLevels.length > 1);
        const removeBtn = isSPMWithOthers ? 
            `<span class="required-badge">Required (Remove other levels first)</span>` :
            `<button type="button" class="btn-remove" onclick="removeEducation(${index})">Remove</button>`;
        
        li.innerHTML = displayText + ' ' + removeBtn;
        itemsList.appendChild(li);
    });
    
    updateFormProgress();
}

// Remove education level from list
function removeEducation(index) {
    const item = educationLevels[index];
    
    // SPM cannot be removed if other education levels exist
    if (item.level === 'SPM' && educationLevels.length > 1) {
        alert('Cannot remove SPM while other education levels exist. Remove other levels first.');
        return;
    }
    
    // If it's SPM and it's the only item, still allow removal but reset the flag
    if (item.level === 'SPM') {
        hasSPM = false;
    }
    
    educationLevels.splice(index, 1);
    updateEducationList();
}

// Handle dynamic form fields based on curricular type
document.getElementById('curricular_type').addEventListener('change', function() {
    const curricularType = this.value;
    const clubFields = document.getElementById('club-fields');
    const activitiesFields = document.getElementById('activities-fields');
    
    // Hide both initially
    clubFields.style.display = 'none';
    activitiesFields.style.display = 'none';
    
    // Show relevant fields
    if (curricularType === 'club') {
        clubFields.style.display = 'block';
    } else if (curricularType === 'activities') {
        activitiesFields.style.display = 'block';
    }
});

// Add club to list
document.getElementById('add-club-btn').addEventListener('click', function() {
    const clubName = document.getElementById('club_name').value.trim();
    const positionHeld = document.getElementById('position_held').value.trim();
    
    if (!clubName || !positionHeld) {
        alert('Please fill in both club name and position held');
        return;
    }
    
    curricularItems.push({
        type: 'club',
        club_name: clubName,
        position_held: positionHeld
    });
    
    updateCurricularList();
    
    // Clear inputs
    document.getElementById('club_name').value = '';
    document.getElementById('position_held').value = '';
});

// Add activity to list
document.getElementById('add-activity-btn').addEventListener('click', function() {
    const activityName = document.getElementById('activity_name').value.trim();
    const activityLevel = document.getElementById('activity_level').value;
    
    if (!activityName || !activityLevel) {
        alert('Please fill in both activity name and level');
        return;
    }
    
    curricularItems.push({
        type: 'activities',
        activity_name: activityName,
        activity_level: activityLevel
    });
    
    updateCurricularList();
    
    // Clear inputs
    document.getElementById('activity_name').value = '';
    document.getElementById('activity_level').value = '';
});

// Update the displayed list
function updateCurricularList() {
    const listContainer = document.getElementById('curricular-list');
    const itemsList = document.getElementById('curricular-items');
    
    if (curricularItems.length === 0) {
        listContainer.style.display = 'none';
        updateFormProgress();
        return;
    }
    
    listContainer.style.display = 'block';
    itemsList.innerHTML = '';
    
    curricularItems.forEach((item, index) => {
        const li = document.createElement('li');
        if (item.type === 'club') {
            li.innerHTML = `<strong>CLUB</strong> ${item.club_name}  (${item.position_held})
                <button type="button" class="btn-remove" onclick="removeItem(${index})">Remove</button>`;
        } else {
            li.innerHTML = `<strong>ACTIVITY</strong> ${item.activity_name} (${item.activity_level})
                <button type="button" class="btn-remove" onclick="removeItem(${index})">Remove</button>`;
        }
        itemsList.appendChild(li);
    });
    
    // Update progress when items change
    updateFormProgress();
}

// Remove item from list
function removeItem(index) {
    curricularItems.splice(index, 1);
    updateCurricularList();
}

document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();    

    // Check if education levels have been added
    if (educationLevels.length === 0) {
        alert('Please add at least your SPM result to proceed');
        return;
    }
    
    // Check if SPM is in the education history
    const hasSPMInList = educationLevels.some(edu => edu.level === 'SPM');
    if (!hasSPMInList) {
        alert('SPM result is required in your education history');
        return;
    }
    
    // Check if at least one curricular item is added
    if (curricularItems.length === 0) {
        alert('Please add at least one club or activity');
        return;
    }
    
    // Show loading state
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    submitBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    
    //Dapatkan value dari form inputs
    const formData = {
        age: parseInt(document.getElementById('age').value),
        race: document.getElementById('race').value,
        field_of_study: document.getElementById('field_of_study').value,
        household_income: parseFloat(document.getElementById('household_income').value),
        education_levels: educationLevels,  // Send all education levels
        curricular_items: curricularItems
    };       

    try {
        //Load model api utk dptkan response drpd AI model
        const response = await fetch('http://localhost:8000/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            //Convert formData to JSON string sebelum prediction
            body: JSON.stringify(formData)
        });

        const result = await response.json();
        const resultDiv = document.getElementById('result');
        
        if (response.ok) {
            // Filter to show only eligible scholarships
            const eligibleScholarships = result.scholarships.filter(s => s.eligibility_status === 'Eligible');

            if (eligibleScholarships.length === 0) {
                resultDiv.innerHTML = `
                    <div class="prediction-result not-eligible">
                        <h4 class="scholarships-title">No Eligible Scholarships Found</h4>
                        <p class="scholarships-subtitle">Based on the criteria (average probability ≥ 30%), you are not currently eligible for any scholarships. Consider improving your academic performance or co-curricular activities.</p>
                    </div>
                `;
                document.getElementById('result-modal').style.display = 'block';
                submitBtn.disabled = false;
                btnText.style.display = 'inline';
                btnLoader.style.display = 'none';
                return;
            }

            // Generate scholarship cards with probability bars
            const scholarshipCards = eligibleScholarships.map(scholarship => {
                // Check if model predictions are available (hybrid and baseline probabilities)
                const hasModelPredictions = scholarship.hybrid_probability && scholarship.baseline_probability;
                
                // Use chosen probability from backend
                const chosenProb = scholarship.chosen_probability || (scholarship.probability * 100);
                const chosenModel = scholarship.chosen_model || 'Unknown';
                
                // Generate model breakdown section if available
                const modelBreakdown = hasModelPredictions ? `
                    <div class="model-breakdown">
                        <div class="model-detail ${chosenModel === 'Hybrid' ? 'chosen' : ''}">
                            <span class="model-label">🤖 Hybrid Model:</span>
                            <span class="model-value">${scholarship.hybrid_probability}</span>
                            <span class="eligibility-badge ${scholarship.hybrid_eligibility === 'Eligible' ? 'eligible' : 'ineligible'}">${scholarship.hybrid_eligibility}</span>
                        </div>
                        <div class="model-detail ${chosenModel === 'Baseline' ? 'chosen' : ''}">
                            <span class="model-label">📊 Baseline Model:</span>
                            <span class="model-value">${scholarship.baseline_probability}</span>
                            <span class="eligibility-badge ${scholarship.baseline_eligibility === 'Eligible' ? 'eligible' : 'ineligible'}">${scholarship.baseline_eligibility}</span>
                        </div>
                        <div class="model-detail average">
                            <span class="model-label">📈 Average:</span>
                            <span class="model-value">${scholarship.average_probability}</span>
                        </div>
                        <div class="chosen-model-info">✨ Using <strong>${chosenModel}</strong> model result (${chosenProb.toFixed(2)}%)</div>
                    </div>
                ` : '';
                
                return `
                    <div class="scholarship-probability-card with-model-predictions">
                        <div class="scholarship-header">
                            <h4>${scholarship.name}</h4>
                            <span class="eligibility eligible">✓ Eligible</span>
                        </div>
                        
                        <p class="scholarship-description">${scholarship.description}</p>
                        
                        ${modelBreakdown}
                        
                        <div class="probability-container">
                            <div class="probability-label">
                                <span>Success Probability
                                </span>
                                <span class="probability-value">${chosenProb.toFixed(2)}%</span>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-fill" style="width: ${chosenProb.toFixed(2)}%"></div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');

            resultDiv.innerHTML = `
                <div class="prediction-result eligible">
                    <h4 class="scholarships-title">
                        🎓 Eligible Scholarships Found (${eligibleScholarships.length})
                    </h4>
                    <p class="scholarships-subtitle">
                        Based on predictions from both Hybrid (Fuzzy Logic + Gradient Boosting) and Baseline (Elastic Net) models. Only scholarships with average probability ≥ 30% are shown.
                    </p>
                    <div class="scholarships-grid">
                        ${scholarshipCards}
                    </div>
                </div>
            `;
            
            // Show the modal
            document.getElementById('result-modal').style.display = 'block';
        } else {
            resultDiv.innerHTML = '<p class="error">Failed to check eligibility. Please try again.</p>';
            document.getElementById('result-modal').style.display = 'block';
        }
    } catch (error) {
        document.getElementById('result').innerHTML = '<p class="error">Error: ' + error.message + '</p>';
        document.getElementById('result-modal').style.display = 'block';
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoader.style.display = 'none';
    }
});

// Close modal when clicking the X button
document.querySelector('.close-modal').addEventListener('click', function() {
    document.getElementById('result-modal').style.display = 'none';
});

// Close modal when clicking outside the modal content
window.addEventListener('click', function(event) {
    const modal = document.getElementById('result-modal');
    if (event.target === modal) {
        modal.style.display = 'none';
    }
});