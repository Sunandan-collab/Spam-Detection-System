document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const emailContent = document.getElementById('email-content');
    const resultsContent = document.getElementById('results-content');
    const loadingElement = document.getElementById('loading');
    const noResultsElement = document.getElementById('no-results');
    const spamIcon = document.getElementById('spam-icon');
    const hamIcon = document.getElementById('ham-icon');
    const resultHeading = document.getElementById('result-heading');
    const resultExplanation = document.getElementById('result-explanation');
    
    let probabilityChart = null;
    
    // Initialize chart
    function initChart(spamProb, hamProb) {
        // If chart already exists, destroy it
        if (probabilityChart) {
            probabilityChart.destroy();
        }
        
        const ctx = document.getElementById('probability-chart').getContext('2d');
        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Spam', 'Ham (Legitimate)'],
                datasets: [{
                    label: 'Probability',
                    data: [spamProb, hamProb],
                    backgroundColor: [
                        'rgba(255, 84, 0, 0.6)',
                        'rgba(56, 176, 0, 0.6)'
                    ],
                    borderColor: [
                        'rgba(255, 84, 0, 1)',
                        'rgba(56, 176, 0, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Probability: ' + (context.raw * 100).toFixed(2) + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Analyze text function
    function analyzeText() {
        const text = emailContent.value.trim();
        
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        // Show loading
        resultsContent.classList.add('hidden');
        noResultsElement.classList.add('hidden');
        loadingElement.classList.remove('hidden');
        
        // Make API request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading
            loadingElement.classList.add('hidden');
            
            // Update results
            updateResults(data);
            
            // Show results
            resultsContent.classList.remove('hidden');
        })
        .catch(error => {
            console.error('Error:', error);
            loadingElement.classList.add('hidden');
            noResultsElement.classList.remove('hidden');
            alert('An error occurred during analysis. Please try again.');
        });
    }
    
    // Update results in the UI
    function updateResults(data) {
        const isSpam = data.is_spam;
        const spamProb = data.spam_probability;
        const hamProb = data.ham_probability;
        
        // Update icons
        spamIcon.classList.toggle('hidden', !isSpam);
        hamIcon.classList.toggle('hidden', isSpam);
        
        // Update text
        if (isSpam) {
            resultHeading.textContent = 'Spam Detected';
            resultHeading.style.color = '#ff5400';
            resultExplanation.textContent = `This message has been classified as spam with ${(spamProb * 100).toFixed(2)}% confidence.`;
        } else {
            resultHeading.textContent = 'Legitimate Message';
            resultHeading.style.color = '#38b000';
            resultExplanation.textContent = `This message appears to be legitimate with ${(hamProb * 100).toFixed(2)}% confidence.`;
        }
        
        // Update chart
        initChart(spamProb, hamProb);
    }
    
    // Clear function
    function clearText() {
        emailContent.value = '';
        resultsContent.classList.add('hidden');
        noResultsElement.classList.remove('hidden');
    }
    
    // Event listeners
    analyzeBtn.addEventListener('click', analyzeText);
    clearBtn.addEventListener('click', clearText);
    
    // Also allow Enter key to submit
    emailContent.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && event.ctrlKey) {
            analyzeText();
        }
    });
});