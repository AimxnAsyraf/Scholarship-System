document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    //Dapatkan value dari form inputs
    const formData = {
        gpa: parseFloat(document.getElementById('gpa').value),
        income: parseFloat(document.getElementById('income').value),
        extracurricular_score: parseInt(document.getElementById('extracurricular_score').value)
    };

    try {
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
            resultDiv.innerHTML = `
                <div class="prediction-result ${result.eligible ? 'eligible' : 'not-eligible'}">
                    <h3>${result.eligible ? '✅ Eligible!' : '❌ Not Eligible'}</h3>
                    <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
                    <h4>Recommended Scholarships:</h4>
                    <ul>
                        ${result.recommended_scholarships.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                    <a href="/apply" class="btn btn-primary">Apply Now</a>
                </div>
            `;
        } else {
            resultDiv.innerHTML = '<p class="error">Failed to check eligibility. Please try again.</p>';
        }
    } catch (error) {
        document.getElementById('result').innerHTML = '<p class="error">Error: ' + error.message + '</p>';
    }
});