let pyodide = null;
let globalDataFrame = null;

// Initialize Pyodide
async function initPyodide() {
    pyodide = await loadPyodide();
    await pyodide.loadPackage(['pandas', 'matplotlib']);
    await setupPythonEnvironment();
}

async function setupPythonEnvironment() {
    await pyodide.runPythonAsync(`
        import pandas as pd
        import matplotlib.pyplot as plt
        from io import StringIO
        
        def clean_data(df):
            # Convert sales/revenue columns to numeric
            numeric_columns = df.select_dtypes(include=['object']).columns
            for col in numeric_columns:
                if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'price', 'amount']):
                    df[col] = pd.to_numeric(df[col].str.replace('[$,]', '', regex=True), errors='coerce')
            
            # Handle missing values
            df = df.fillna(df.mean(numeric_only=True))
            return df
            
        def analyze_sales(df):
            results = {}
            
            # Basic statistics
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            sales_col = [col for col in numeric_cols if 'sale' in col.lower() or 'revenue' in col.lower()][0]
            
            results['total_sales'] = df[sales_col].sum()
            results['average_sales'] = df[sales_col].mean()
            
            # Best performing products/categories
            if 'category' in df.columns.str.lower():
                cat_col = df.columns[df.columns.str.lower() == 'category'][0]
                top_categories = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False).head(5)
                results['top_categories'] = top_categories.to_dict()
            
            # Monthly trends
            if any('date' in col.lower() for col in df.columns):
                date_col = [col for col in df.columns if 'date' in col.lower()][0]
                df[date_col] = pd.to_datetime(df[date_col])
                monthly_sales = df.groupby(df[date_col].dt.strftime('%Y-%m'))[sales_col].sum()
                results['monthly_sales'] = monthly_sales.to_dict()
            
            return results
    `);
}

// Initialize on page load
window.addEventListener('load', initPyodide);

// Handle file upload
document.getElementById('fileInput').addEventListener('change', function(e) {
    document.getElementById('analyzeBtn').disabled = !e.target.files.length;
});

document.getElementById('analyzeBtn').addEventListener('click', async function() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    try {
        document.getElementById('loadingIndicator').classList.remove('d-none');
        document.getElementById('results').classList.add('d-none');
        
        const text = await file.text();
        
        // Load and process data with Python
        const result = await pyodide.runPythonAsync(`
            df = pd.read_csv(StringIO('''${text}'''))
            df = clean_data(df)
            analyze_sales(df)
        `);
        
        const analysisResults = result.toJs();
        displayResults(analysisResults);
        createCharts(analysisResults);
        
        document.getElementById('results').classList.remove('d-none');
    } catch (error) {
        alert('Error processing file: ' + error.message);
    } finally {
        document.getElementById('loadingIndicator').classList.add('d-none');
    }
});

function displayResults(results) {
    const statsDiv = document.getElementById('summaryStats');
    statsDiv.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h5>Total Sales</h5>
                <p>$${results.total_sales.toLocaleString('en-US', {maximumFractionDigits: 2})}</p>
            </div>
            <div class="col-md-6">
                <h5>Average Sales</h5>
                <p>$${results.average_sales.toLocaleString('en-US', {maximumFractionDigits: 2})}</p>
            </div>
        </div>
    `;
}

function createCharts(results) {
    // Monthly Sales Trend Chart
    if (results.monthly_sales) {
        const labels = Object.keys(results.monthly_sales);
        const data = Object.values(results.monthly_sales);
        
        new Chart(document.getElementById('salesTrendChart'), {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Monthly Sales',
                    data: data,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: value => '$' + value.toLocaleString()
                        }
                    }
                }
            }
        });
    }
    
    // Category Performance Chart
    if (results.top_categories) {
        const labels = Object.keys(results.top_categories);
        const data = Object.values(results.top_categories);
        
        new Chart(document.getElementById('categoryChart'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Sales by Category',
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgb(54, 162, 235)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: value => '$' + value.toLocaleString()
                        }
                    }
                }
            }
        });
    }
}

// PDF Report Generation
document.getElementById('downloadReport').addEventListener('click', async function() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    // Add title
    doc.setFontSize(20);
    doc.text('Sales Analysis Report', 20, 20);
    
    // Add summary statistics
    doc.setFontSize(14);
    doc.text('Summary Statistics', 20, 40);
    const summaryStats = document.getElementById('summaryStats').innerText;
    doc.setFontSize(12);
    doc.text(summaryStats, 20, 50);
    
    // Add charts
    const charts = document.querySelectorAll('canvas');
    let yPosition = 80;
    
    charts.forEach((canvas, index) => {
        if (yPosition + 100 > doc.internal.pageSize.height) {
            doc.addPage();
            yPosition = 20;
        }
        
        doc.text(canvas.closest('.card').querySelector('h4').innerText, 20, yPosition);
        doc.addImage(canvas.toDataURL(), 'PNG', 20, yPosition + 10, 170, 80);
        yPosition += 100;
    });
    
    // Save the PDF
    doc.save('sales-analysis-report.pdf');
});