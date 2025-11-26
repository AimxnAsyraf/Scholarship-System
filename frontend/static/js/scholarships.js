// Fetch scholarships from backend API
fetch('http://localhost:8000/api/scholarships')
    .then(response => response.json())
    .then(data => {
        const container = document.getElementById('scholarships-list');
        container.innerHTML = '';
        
        data.forEach(scholarship => {
            const card = document.createElement('div');
            card.className = 'scholarship-card';
            card.innerHTML = `
                <h3>${scholarship.name}</h3>
                <p class="amount">$${scholarship.amount.toFixed(2)}</p>
                <p>${scholarship.description}</p>
                <a href="/apply" class="btn btn-primary">Apply Now</a>
            `;
            container.appendChild(card);
        });
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('scholarships-list').innerHTML = 
            '<p class="error">Failed to load scholarships. Please try again later.</p>';
    });