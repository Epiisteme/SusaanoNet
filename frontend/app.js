// Global Chart Instances
let charts = { BTC: null, ETH: null, SOL: null };
let latestData = null; // Store latest forecasts and history for instant horizon toggling

// Common Chart Configuration
const getChartConfig = (assetName, data, borderColor, labels) => ({
    type: 'line',
    data: {
        labels: labels,
        datasets: [
            {
                label: 'Live Price',
                data: data,
                borderColor: borderColor,
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 1, // smaller points
                pointBackgroundColor: '#fff'
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                grid: { display: false },
                ticks: {
                    maxTicksLimit: 6 
                }
            },
            y: {
                grid: { color: '#F1F5F9' },
                ticks: {
                    callback: function(value) {
                        return '$' + value;
                    }
                }
            }
        }
    }
});

// Initialize Empty Charts
function initCharts() {
    const btcCtx = document.getElementById('btcChart').getContext('2d');
    charts.BTC = new Chart(btcCtx, getChartConfig('BTC', [], '#1E293B', []));

    const ethCtx = document.getElementById('ethChart').getContext('2d');
    charts.ETH = new Chart(ethCtx, getChartConfig('ETH', [], '#1E293B', []));

    const solCtx = document.getElementById('solChart').getContext('2d');
    charts.SOL = new Chart(solCtx, getChartConfig('SOL', [], '#1E293B', []));
}

// Format Currency
const formatMoney = (num) => {
    return '$' + num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

// Update UI based on currently selected horizon and globally stored latestData
function updateDashboardUI() {
    if (!latestData) return;
    
    const horizon = document.getElementById('horizon-selector').value;
    const history = latestData.history;
    const forecasts = latestData.forecasts;
    
    ['BTC', 'ETH', 'SOL'].forEach(asset => {
        const result = forecasts[asset];
        const liveHistory = history[asset];
        const liveTimestamps = history.timestamps;
        
        // Dynamically get the data for the selected horizon
        const q_data = result[`quantiles_${horizon}`];
        const d_data = result[`direction_${horizon}`];
        
        // Update Text Metrics (Quantiles)
        document.getElementById(`${asset.toLowerCase()}-p90`).textContent = formatMoney(q_data.p90_breakout_boundary);
        document.getElementById(`${asset.toLowerCase()}-p50`).textContent = formatMoney(q_data.p50_median_forecast);
        document.getElementById(`${asset.toLowerCase()}-p10`).textContent = formatMoney(q_data.p10_crash_boundary);
        document.getElementById(`${asset.toLowerCase()}-price`).textContent = formatMoney(liveHistory[liveHistory.length - 1]);

        // Update Directional Badge
        const badge = document.getElementById(`${asset.toLowerCase()}-dir`);
        badge.style.display = 'inline-flex';
        badge.className = 'dir-badge'; // reset classes
        
        if (d_data.direction === 'up') {
            badge.classList.add('up');
            badge.textContent = `UP`;
        } else if (d_data.direction === 'down') {
            badge.classList.add('down');
            badge.textContent = `DOWN`;
        } else {
            badge.classList.add('neutral');
            badge.textContent = `NEUTRAL`;
        }

        // Update Chart
        const chart = charts[asset];
        const newLabels = [...liveTimestamps, `T+${horizon} (Forecast)`];
        chart.data.labels = newLabels;

        chart.data.datasets = [
            {
                label: 'Live Price',
                data: [...liveHistory, q_data.p50_median_forecast], // main line continues to P50
                borderColor: '#3B82F6', 
                backgroundColor: 'transparent',
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 1,
                pointBackgroundColor: '#fff'
            },
            {
                label: 'Upper Confidence Bound',
                // Null pad the history so the dot only appears at the end
                data: [...Array(liveHistory.length).fill(null), q_data.p90_breakout_boundary],
                borderColor: '#10B981',
                borderDash: [5, 5],
                pointBackgroundColor: '#10B981',
                pointRadius: 6,
                fill: false
            },
            {
                label: 'Lower Confidence Bound',
                data: [...Array(liveHistory.length).fill(null), q_data.p10_crash_boundary],
                borderColor: '#EF4444',
                borderDash: [5, 5],
                pointBackgroundColor: '#EF4444',
                pointRadius: 6,
                fill: false
            }
        ];

        chart.update();
    });
}

async function fetchLiveQuantiles() {
    const btn = document.getElementById('predict-btn');
    const loader = document.getElementById('btn-loader');
    const btnText = document.querySelector('.btn-text');

    // UI Loading State
    btn.disabled = true;
    btnText.textContent = "Scraping Live Binance Data...";
    loader.classList.remove('hidden');

    try {
        // Hit the live production endpoint
        const response = await fetch('http://localhost:8000/predict/live', {
            method: 'GET'
        });
        
        if (!response.ok) {
            throw new Error(`API returned status ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save the full 3-horizon payload to global state
        latestData = data;
        
        // Update UI based on whatever horizon is currently selected
        updateDashboardUI();

        btnText.textContent = "Live Market Sync Successful!";
        setTimeout(() => {
            btnText.textContent = "Refresh Quantile Forecasts";
        }, 3000);

    } catch (error) {
        console.error("Prediction Error:", error);
        alert("Failed to connect to Live API. Ensure FastAPI is running on port 8000.");
        btnText.textContent = "Generate Quantile Forecasts";
    } finally {
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    
    // Fetch data on button click
    document.getElementById('predict-btn').addEventListener('click', fetchLiveQuantiles);
    
    // Instantly update UI when horizon dropdown is changed
    document.getElementById('horizon-selector').addEventListener('change', updateDashboardUI);
});
