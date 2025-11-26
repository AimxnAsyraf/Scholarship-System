// Store curricular items
let curricularItems = [];

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
    
    // Clear required attributes
    document.getElementById('spm_as').removeAttribute('required');
    document.getElementById('cgpa').removeAttribute('required');
    document.getElementById('alevel_stars').removeAttribute('required');
    
    // Show relevant field and set required
    if (level === 'SPM') {
        spmField.style.display = 'block';
        document.getElementById('spm_as').setAttribute('required', 'required');
    } else if (level === 'A Level') {
        alevelField.style.display = 'block';
        document.getElementById('alevel_stars').setAttribute('required', 'required');
    } else if (level === 'STPM' || level === 'Matriculation' || level === 'Foundation' || level === 'Undergraduate') {
        cgpaField.style.display = 'block';
        document.getElementById('cgpa').setAttribute('required', 'required');
    }
});

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
            li.innerHTML = `<strong>Club:</strong> ${item.club_name} - ${item.position_held} 
                <button type="button" class="btn-remove" onclick="removeItem(${index})">Remove</button>`;
        } else {
            li.innerHTML = `<strong>Activity:</strong> ${item.activity_name} (${item.activity_level}) 
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
    const levelOfStudy = document.getElementById('level_of_study').value;
    const formData = {
        age: parseInt(document.getElementById('age').value),
        race: document.getElementById('race').value,
        level_of_study: levelOfStudy,
        field_of_study: document.getElementById('field_of_study').value,
        household_income: parseFloat(document.getElementById('household_income').value),
        curricular_items: curricularItems
    };
    
    // Add appropriate academic performance field based on level
    if (levelOfStudy === 'SPM') {
        formData.spm_as = parseInt(document.getElementById('spm_as').value);
    } else if (levelOfStudy === 'A Level') {
        formData.alevel_stars = parseInt(document.getElementById('alevel_stars').value);
    } else {
        formData.cgpa = parseFloat(document.getElementById('cgpa').value);
    }

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
            // Generate scholarship cards with probability bars
            const scholarshipCards = result.scholarships.map(scholarship => `
                <div class="scholarship-probability-card">
                    <div class="scholarship-header">
                        <h4>${scholarship.name}</h4>
                        <span class="eligibility ${scholarship.eligibility_status === 'Eligible' ? 'eligible' : 'not-eligible'}">${scholarship.eligibility_status}</span>
                    </div>
                    
                    <p class="scholarship-description">${scholarship.description}</p>
                    <div class="probability-container">
                        <div class="probability-label">
                            <span>Eligibility Probability</span>
                            <span class="probability-value">${(scholarship.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${(scholarship.probability * 100).toFixed(1)}%"></div>
                        </div>
                    </div>
                </div>
            `).join('');

            resultDiv.innerHTML = `
                <div class="prediction-result ${result.eligible ? 'eligible' : 'not-eligible'}">
                    <div class="result-header">
                        <h3>${result.eligible ? '✅ Congratulations!' : '❌ Not Currently Eligible'}</h3>
                        <p class="status-badge">${result.eligibility_status}</p>
                    </div>
                    <div class="result-stats">
                        <div class="stat-item">
                            <span class="stat-label">Prediction Score</span>
                            <span class="stat-value">${result.prediction_score.toFixed(2)} / 2.00</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Model Confidence</span>
                            <span class="stat-value">${(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                    </div>
                    <h4 class="scholarships-title">Scholarship Eligibility Probabilities</h4>
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