// Dashboard Orchestration Logic

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const loadingContainer = document.getElementById('loadingContainer');
    const errorContainer = document.getElementById('errorContainer');
    const errorMessage = document.getElementById('errorMessage');
    const dashboardContent = document.getElementById('dashboardContent');
    const submitBtn = document.getElementById('submitBtn');
    
    // Metrics DOM
    const metricClose = document.getElementById('metricClose');
    const metricLstm = document.getElementById('metricLstm');
    const metricRf = document.getElementById('metricRf');
    const metricVolatility = document.getElementById('metricVolatility');
    const metricLstmRoi = document.getElementById('metricLstmRoi');
    const metricRfRoi = document.getElementById('metricRfRoi');
    const metricHigh = document.getElementById('metricHigh');
    const metricLow = document.getElementById('metricLow');

    let chartInstance = null;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const ticker = document.getElementById('ticker').value.trim().toUpperCase();
        const companyName = document.getElementById('companyName').value.trim();

        if (!ticker || !companyName) return;

        // Reset UI State
        errorContainer.classList.add('hidden');
        dashboardContent.classList.add('hidden');
        loadingContainer.classList.remove('hidden');
        submitBtn.disabled = true;
        submitBtn.classList.add('opacity-50', 'cursor-not-allowed');

        try {
            // Dynamic API Root Gateway Isolation
            const API_BASE_URL = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
              ? 'http://127.0.0.1:8000'
              : window.location.origin;

            // Network API Request Handler
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ticker: ticker,
                    company_name: companyName,
                    horizon: 90
                })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Server Execution Error Traceout Context:", errorText);
                throw new Error(`Server returned error status payload code: ${response.status}`);
            }

            const data = await response.json();
            
            // Update UI Metrics
            updateMetrics(data);
            
            // Render Statistical Visualization Engine
            renderChart(data);

            // Show Dashboard
            loadingContainer.classList.add('hidden');
            dashboardContent.classList.remove('hidden');

        } catch (error) {
            console.error('Fetch Error:', error);
            errorMessage.textContent = error.message || 'Failed to fetch prediction data. Please check connection to backend.';
            loadingContainer.classList.add('hidden');
            errorContainer.classList.remove('hidden');
        } finally {
            submitBtn.disabled = false;
            submitBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    });

    function updateMetrics(data) {
        const lastClose = data.historical_prices[data.historical_prices.length - 1];
        const targetLstm = data.lstm_median_path[data.lstm_median_path.length - 1];
        const targetRf = data.rf_median_path[data.rf_median_path.length - 1];
        const rangeLower = data.lower_volatility_bound[data.lower_volatility_bound.length - 1];
        const rangeUpper = data.upper_volatility_bound[data.upper_volatility_bound.length - 1];

        const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
        const formatPercent = (val) => new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(val);

        metricClose.textContent = formatCurrency(lastClose);
        metricLstm.textContent = formatCurrency(targetLstm);
        metricRf.textContent = formatCurrency(targetRf);
        metricVolatility.textContent = `${formatCurrency(rangeLower)} - ${formatCurrency(rangeUpper)}`;

        const lstmRoi = (targetLstm - lastClose) / lastClose;
        const rfRoi = (targetRf - lastClose) / lastClose;
        const histHigh = Math.max(...data.historical_prices);
        const histLow = Math.min(...data.historical_prices);

        metricLstmRoi.textContent = formatPercent(lstmRoi);
        metricLstmRoi.className = lstmRoi >= 0 ? "metric-sub accent-pos" : "metric-sub accent-neg";
        
        metricRfRoi.textContent = formatPercent(rfRoi);
        metricRfRoi.className = rfRoi >= 0 ? "metric-sub accent-pos" : "metric-sub accent-neg";

        metricHigh.textContent = formatCurrency(histHigh);
        metricLow.textContent = formatCurrency(histLow);
    }

    function renderChart(data) {
        const ctx = document.getElementById('predictionChart').getContext('2d');

        // State Management - Destroy old chart instance
        if (chartInstance) {
            chartInstance.destroy();
        }

        // Concatenate timelines
        const labels = [...data.historical_dates, ...data.dates];
        
        // Pad future arrays with nulls for historical period so they align correctly
        const padHistory = new Array(data.historical_dates.length).fill(null);
        // Connect the last historical point to the future paths
        const connectHistory = new Array(data.historical_dates.length - 1).fill(null);
        connectHistory.push(data.historical_prices[data.historical_prices.length - 1]);

        const lstmPath = [...connectHistory, ...data.lstm_median_path.slice(1)];
        const rfPath = [...connectHistory, ...data.rf_median_path.slice(1)];
        const upperPath = [...connectHistory, ...data.upper_volatility_bound.slice(1)];
        const lowerPath = [...connectHistory, ...data.lower_volatility_bound.slice(1)];
        
        // Historical path padded with nulls for future
        const padFuture = new Array(data.dates.length).fill(null);
        const historicalPath = [...data.historical_prices, ...padFuture];

        // Chart.js Configuration Matrix
        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Upper Volatility Bound (95%)',
                        data: upperPath,
                        borderColor: 'rgba(155, 35, 53, 0.35)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Lower Volatility Bound (5%)',
                        data: lowerPath,
                        borderColor: 'rgba(155, 35, 53, 0.35)',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        fill: '-1',
                        backgroundColor: 'rgba(155, 35, 53, 0.04)',
                        tension: 0.1
                    },
                    {
                        label: 'Historical Spot Price',
                        data: historicalPath,
                        borderColor: '#1A1410',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'LSTM Forecast',
                        data: lstmPath,
                        borderColor: '#C8A24E',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Random Forest Forecast',
                        data: rfPath,
                        borderColor: '#5C3D2E',
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#8C7B6E',
                            usePointStyle: true,
                            pointStyleWidth: 16,
                            font: { family: "'JetBrains Mono', monospace", size: 11 },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1A1410',
                        titleColor: '#F9F7F2',
                        bodyColor: '#DDD8CF',
                        borderColor: '#4A3F35',
                        borderWidth: 1,
                        padding: 12,
                        titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                        bodyFont: { family: "'Hanken Grotesk', sans-serif", size: 13 },
                        callbacks: {
                            label: (ctx) => ` ${ctx.dataset.label}: $${Number(ctx.raw).toFixed(2)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#DDD8CF', drawBorder: false },
                        ticks: {
                            color: '#8C7B6E',
                            maxTicksLimit: 10,
                            font: { family: "'JetBrains Mono', monospace", size: 10 }
                        }
                    },
                    y: {
                        grid: { color: '#DDD8CF', drawBorder: false },
                        ticks: {
                            color: '#8C7B6E',
                            font: { family: "'JetBrains Mono', monospace", size: 10 },
                            callback: (v) => '$' + v
                        }
                    }
                }
            }
        });
    }
});
