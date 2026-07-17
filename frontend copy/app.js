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
        loadingContainer.classList.add('flex');
        submitBtn.disabled = true;
        submitBtn.classList.add('opacity-50', 'cursor-not-allowed');

        try {
            // Network API Request Handler
            const response = await fetch('http://127.0.0.1:8000/api/predict', {
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
                const errorData = await response.json().catch(() => null);
                const detail = errorData?.detail || `Server error: ${response.status} ${response.statusText}`;
                throw new Error(detail);
            }

            const data = await response.json();
            
            // Update UI Metrics
            updateMetrics(data);
            
            // Render Statistical Visualization Engine
            renderChart(data);

            // Show Dashboard
            loadingContainer.classList.remove('flex');
            loadingContainer.classList.add('hidden');
            dashboardContent.classList.remove('hidden');
            dashboardContent.classList.add('flex');

        } catch (error) {
            console.error('Fetch Error:', error);
            errorMessage.textContent = error.message || 'Failed to fetch prediction data. Please check connection to backend.';
            loadingContainer.classList.remove('flex');
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
        metricLstmRoi.className = lstmRoi >= 0 ? "text-2xl font-bold text-green-400" : "text-2xl font-bold text-rose-400";
        
        metricRfRoi.textContent = formatPercent(rfRoi);
        metricRfRoi.className = rfRoi >= 0 ? "text-2xl font-bold text-green-400" : "text-2xl font-bold text-rose-400";

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
                        borderColor: 'rgba(239, 68, 68, 0.4)', // light red dashed
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Lower Volatility Bound (5%)',
                        data: lowerPath,
                        borderColor: 'rgba(239, 68, 68, 0.4)', // light red dashed
                        borderWidth: 1,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: '-1', // Fill to the previous dataset (Upper Bound)
                        backgroundColor: 'rgba(239, 68, 68, 0.05)', // ultra-light red opacity layer
                        tension: 0.1
                    },
                    {
                        label: 'Historical Spot Price',
                        data: historicalPath,
                        borderColor: '#ffffff', // High contrast white
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'LSTM Forecast',
                        data: lstmPath,
                        borderColor: '#3b82f6', // Tailwind blue-500
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Random Forest Forecast',
                        data: rfPath,
                        borderColor: '#f97316', // Tailwind orange-500
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
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#94a3b8', // slate-400
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(15, 23, 42, 0.9)', // slate-900
                        titleColor: '#f1f5f9', // slate-100
                        bodyColor: '#cbd5e1', // slate-300
                        borderColor: '#334155', // slate-700
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: '#334155', // slate-700
                            drawBorder: false
                        },
                        ticks: {
                            color: '#94a3b8', // slate-400
                            maxTicksLimit: 10
                        }
                    },
                    y: {
                        grid: {
                            color: '#334155', // slate-700
                            drawBorder: false
                        },
                        ticks: {
                            color: '#94a3b8', // slate-400
                            callback: function(value, index, values) {
                                return '$' + value;
                            }
                        }
                    }
                }
            }
        });
    }
});
