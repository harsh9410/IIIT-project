// ===============================
// 🔌 API Configuration
// ===============================
// Backend API served by FastAPI. Keep separate from static server.
const API_BASE_URL = 'http://localhost:8001';
let ws;
let pollingIntervalId;

// ===============================
// 📊 Initialize Chart
// ===============================
let powerChart;

function initializeChart() {
    const ctx = document.getElementById('powerChart').getContext('2d');
    powerChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['AC', 'Refrigerator', 'Washing Machine', 'Lights', 'Other'],
            datasets: [{
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    '#3b82f6',
                    '#10b981',
                    '#f59e0b',
                    '#ef4444',
                    '#8b5cf6'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            },
            cutout: '65%'
        }
    });
}

// ===============================
// 📈 Fetch Current Reading
// ===============================
async function fetchCurrentReading() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/current-reading`);
        const data = await response.json();
        
        updateDashboard(data);
        updateApplianceBreakdown(data.appliances);
        updateChart(data.appliances);
        checkAlert(data.total_power);
            
    } catch (error) {
        console.warn('Waiting for real-time data...');
        showWaitingState();
    }
}

function updateDashboard(data) {
    document.getElementById('voltageVal').textContent = `${data.voltage} V`;
    document.getElementById('currentVal').textContent = `${data.current} A`;
    document.getElementById('powerVal').textContent = `${data.total_power} W`;
    
    // Add animation
    animateValue('voltageVal');
    animateValue('currentVal');
    animateValue('powerVal');
}

function updateApplianceBreakdown(appliances) {
    document.getElementById('acPower').textContent = `${appliances.AC} W`;
    document.getElementById('fridgePower').textContent = `${appliances.Refrigerator} W`;
    document.getElementById('washPower').textContent = `${appliances['Washing Machine']} W`;
    document.getElementById('lightsPower').textContent = `${appliances.Lights} W`;
    document.getElementById('otherPower').textContent = `${appliances.Other} W`;
    
    // Update usage bars
    updateUsageBars(appliances);
}

function updateUsageBars(appliances) {
    const totalPower = Object.values(appliances).reduce((sum, power) => sum + power, 0);
    
    document.querySelectorAll('.appliance-item').forEach(item => {
        const applianceType = item.classList[1];
        const power = appliances[getApplianceName(applianceType)];
        const percentage = (power / totalPower) * 100;
        
        const usageFill = item.querySelector('.usage-fill');
        usageFill.style.width = `${percentage}%`;
    });
}

function getApplianceName(className) {
    const names = {
        'ac': 'AC',
        'refrigerator': 'Refrigerator',
        'washing': 'Washing Machine',
        'lights': 'Lights',
        'other': 'Other'
    };
    return names[className];
}

function updateChart(appliances) {
    if (powerChart) {
        powerChart.data.datasets[0].data = [
            appliances.AC,
            appliances.Refrigerator,
            appliances['Washing Machine'],
            appliances.Lights,
            appliances.Other
        ];
        powerChart.update();
    }
}

// ===============================
// ⚠️ Alert System
// ===============================
function checkAlert(power) {
    const status = document.getElementById('alertStatus');
    const trend = document.getElementById('statusTrend');
    
    if (power > 500) {
        status.textContent = 'High Usage!';
        status.style.color = '#ef4444';
        trend.textContent = 'Reduce Load';
        trend.style.color = '#ef4444';
    } else if (power > 300) {
        status.textContent = 'Moderate';
        status.style.color = '#f59e0b';
        trend.textContent = 'Stable';
        trend.style.color = '#f59e0b';
    } else {
        status.textContent = 'Normal';
        status.style.color = '#10b981';
        trend.textContent = 'Optimal';
        trend.style.color = '#10b981';
    }
}

// ===============================
// 🔮 Fetch Predictions
// ===============================
async function fetchPredictions() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict?hours=6`);
        const data = await response.json();
        displayPredictions(data.predictions);
    } catch (error) {
        console.error('Error fetching predictions:', error);
        displaySimulatedPredictions();
    }
}

function displayPredictions(predictions) {
    const pred1h = predictions.find(p => {
        const time = new Date(p.timestamp).getTime() - new Date().getTime();
        return time <= 3600000; // 1 hour
    });
    
    const pred3h = predictions[2]; // 3rd hour
    const pred6h = predictions[5]; // 6th hour
    
    if (pred1h) {
        document.getElementById('pred1h').textContent = `${pred1h.predicted_power} W`;
    }
    if (pred3h) {
        document.getElementById('pred3h').textContent = `${pred3h.predicted_power} W`;
    }
    if (pred6h) {
        document.getElementById('pred6h').textContent = `${pred6h.predicted_power} W`;
    }
}

// ===============================
// 📤 CSV Upload & Analysis
// ===============================
document.getElementById('csvFile').addEventListener('change', function(e) {
    const file = e.target.files[0];
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadArea = document.getElementById('uploadArea');
    
    if (file) {
        analyzeBtn.disabled = false;
        uploadArea.innerHTML = `
            <i class="fas fa-check-circle" style="color: #10b981;"></i>
            <h3>File Selected</h3>
            <p>${file.name}</p>
            <small>Ready for CNN analysis</small>
        `;
    }
});

document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const fileInput = document.getElementById('csvFile');
    if (!fileInput.files.length) return;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/upload-csv`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        showAnalysisResult(result);
        
    } catch (error) {
        console.error('Upload error:', error);
        alert('Error analyzing file. Please try again.');
    } finally {
        hideLoading();
    }
});

function showAnalysisResult(result) {
    alert(`CNN Analysis Complete!\n\nProcessed ${result.rows_processed} rows\nAccuracy: ${(result.analysis.prediction_accuracy * 100).toFixed(1)}%`);
    
    // Update dashboard with analysis results
    if (result.analysis.appliance_breakdown) {
        updateApplianceBreakdown(result.analysis.appliance_breakdown);
        updateChart(result.analysis.appliance_breakdown);
    }
}

// ===============================
// 🎬 Animations
// ===============================
function animateValue(elementId) {
    const element = document.getElementById(elementId);
    element.style.transform = 'scale(1.1)';
    setTimeout(() => {
        element.style.transform = 'scale(1)';
    }, 300);
}

function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showWaitingState() {
    document.getElementById('voltageVal').textContent = '—';
    document.getElementById('currentVal').textContent = '—';
    document.getElementById('powerVal').textContent = '—';
    document.getElementById('alertStatus').textContent = 'Waiting…';
    document.getElementById('statusTrend').textContent = 'No data yet';
    if (powerChart) {
        powerChart.data.datasets[0].data = [0, 0, 0, 0, 0];
        powerChart.update();
    }
}

function displaySimulatedPredictions() {
    document.getElementById('pred1h').textContent = `${(115 + Math.random() * 30 - 15).toFixed(1)} W`;
    document.getElementById('pred3h').textContent = `${(120 + Math.random() * 40 - 20).toFixed(1)} W`;
    document.getElementById('pred6h').textContent = `${(125 + Math.random() * 50 - 25).toFixed(1)} W`;
}

// ===============================
// 🚀 Initialize Application
// ===============================
document.addEventListener('DOMContentLoaded', function() {
    initializeChart();
    // Start with polling so UI updates even if WS is delayed/blocked
    startPolling();
    // Try WebSocket; if it connects, polling will auto-stop
    setupLiveUpdates();
    fetchPredictions();
    
    // Update predictions every minute
    setInterval(fetchPredictions, 60000);
});

function setupLiveUpdates() {
    const wsProtocol = API_BASE_URL.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${new URL(API_BASE_URL).host}/ws/live`;
    try {
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
            // Stop polling if it was running
            if (pollingIntervalId) {
                clearInterval(pollingIntervalId);
                pollingIntervalId = null;
            }
        };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
            updateApplianceBreakdown(data.appliances);
            updateChart(data.appliances);
            checkAlert(data.total_power);
        };
        ws.onclose = () => {
            // Fallback to polling every 5s
            startPolling();
        };
        ws.onerror = () => {
            try { ws.close(); } catch (e) {}
            startPolling();
        };
    } catch (e) {
        startPolling();
    }
}

function startPolling() {
    fetchCurrentReading();
    if (!pollingIntervalId) {
        pollingIntervalId = setInterval(fetchCurrentReading, 5000);
    }
}

// ===============================
// 🎯 Smooth Scrolling
// ===============================
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);
        
        if (targetElement) {
            targetElement.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});