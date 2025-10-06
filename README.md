# 🚀 Trader Performance vs Market Sentiment Analysis

## 📖 Project Overview

This project presents a comprehensive data analysis exploring how **Bitcoin market sentiment (Fear/Greed Index)** affects **trader performance** using historical trading data from Hyperliquid. The analysis uncovers correlations, behavioral patterns, and actionable insights for smarter trading strategies under different sentiment conditions.

---

## 📊 Major Findings

### 🏆 **Primary Discovery**

**Greed periods significantly outperform Fear periods** with statistically validated evidence:

| Metric                       | Fear Periods | Greed Periods | Improvement            |
| ---------------------------- | ------------ | ------------- | ---------------------- |
| **Average PnL/Trade**        | $50.05       | $77.84        | **+55.5%**             |
| **Win Rate**                 | 41.5%        | 45.4%         | **+3.9pp**             |
| **Statistical Significance** | p < 0.000001 | p < 0.000001  | **Highly Significant** |

### 📈 **Dataset Statistics**

- **184,263** trading records with matched sentiment data
- **32** unique trader accounts
- **Date Range**: March 28, 2023 to February 19, 2025
- **Total Volume**: $880,912,169.43
- **Confidence Level**: 99.9999% (p < 0.000001)

---

## 🏗️ Project Structure

```
trader_sentiment_analysis/
│
├── data/                           # Data files
│   ├── trader_data.csv            # Original Hyperliquid data
│   ├── sentiment_data.csv         # Bitcoin Fear/Greed Index data
│   ├── merged_trader_sentiment_data.csv  # Main analysis dataset
│   └── account_performance_metrics.csv   # Account-level statistics
│
├── src/                           # Source code modules
│   ├── data_utils.py             # Data loading and preprocessing
│   ├── preprocessing.py          # Data cleaning and feature engineering
│   ├── visualization.py          # Chart generation functions
│   └── analysis.py               # Statistical testing functions
│
├── notebooks/                    # Jupyter notebooks
│   |── 01_complete_analysis.ipynb  # Comprehensive analysis notebook
|   |── analysis_simple.ipynb
│
├── reports/                      # Generated outputs
│   ├── summary_findings.md       # Executive summary report
│   ├── interactive_dashboard.html # Interactive Plotly dashboard
│   ├── pnl_distribution_by_sentiment.png
│   ├── leverage_analysis.png
│   ├── time_series_analysis.png
│   ├── performance_heatmap.png
│   └── win_rate_analysis.png
│
├── requirements.txt              # Python dependencies
├── download_data.py             # Data download script
└── README.md                    # This file
```

---

## 🚀 How to Run This Project

### 1. 🔧 Environment Setup

```powershell
# Navigate to the project directory
cd C:\Users\hi\trader_sentiment_analysis

# Verify Python version
python --version  # Should show Python 3.7+

# Install all required dependencies
pip install -r requirements.txt
```

### 2. 📥 Download & Load Data

```powershell
# Download datasets from Google Drive (this may take a few minutes)
python download_data.py
```

### 3. 🔄 Data Preprocessing

```powershell
# Run the complete preprocessing pipeline
python src\preprocessing.py
```

### 4. 📊 Generate Analysis & Visualizations

#### Option A: Run Complete Analysis (Recommended)

```powershell
# Generate all visualizations and statistical analysis
python -c "import pandas as pd; import sys; sys.path.append('src'); from analysis import run_comprehensive_analysis; from visualization import save_all_plots; df = pd.read_csv('data/merged_trader_sentiment_data.csv'); analysis_results = run_comprehensive_analysis(df); plot_files = save_all_plots(df)"
```

### 5. 🎨 Launch Jupyter Notebook

#### Simple Notebook (Recommended)

```powershell
# Launch the simplified, working notebook
jupyter notebook notebooks\\analysis_simple.ipynb
```

### 6. 🌐 View Interactive Dashboard

```powershell
# Open the interactive dashboard in your default browser
start reports\interactive_dashboard.html
```

### 7. 📋 Review Results

- **📓 Complete Analysis**: Open `notebooks/01_complete_analysis.ipynb` in Jupyter
- **📊 Interactive Dashboard**: Open `reports/interactive_dashboard.html` in browser
- **📋 Executive Summary**: Read `reports/summary_findings.md`
- **🖼️ Static Visualizations**: View PNG files in `reports/` folder

### 8. ✅ **Quick Verification**

To verify everything worked correctly, check that these files exist:

```powershell
# Check all expected files are generated
ls data/
# Expected: trader_data.csv, sentiment_data.csv, merged_trader_sentiment_data.csv, account_performance_metrics.csv

ls reports/
# Expected: 5 PNG files + 1 HTML file + summary_findings.md

ls notebooks/
# Expected: 01_complete_analysis.ipynb, analysis_simple.ipynb, .ipynb_checkpoints
```

**Expected File Counts:**

- **Data files**: 4 CSV files in `data/`
- **Visualizations**: 5 PNG files in `reports/`
- **Interactive dashboard**: 1 HTML file in `reports/`
- **Documentation**: 1 Jupyter notebook + 1 summary report

**Quick Validation Command:**

```powershell
python -c "import os; print('Data files:', len([f for f in os.listdir('data') if f.endswith('.csv')])); print('Visualizations:', len([f for f in os.listdir('reports') if f.endswith('.png')])); print('Dashboard:', 'interactive_dashboard.html' in os.listdir('reports')); print('Project complete!' if len([f for f in os.listdir('data') if f.endswith('.csv')]) >= 4 else 'Run preprocessing first')"
```

---

---

## 📊 Visualizations

### 🎯 **Key Charts Generated**

1. **PnL Distribution by Sentiment** - Comprehensive profit/loss analysis
2. **Leverage Analysis** - Risk behavior patterns across sentiment phases
3. **Time Series Analysis** - Temporal performance trends
4. **Correlation Heatmap** - Relationship matrix between metrics
5. **Win Rate Analysis** - Success rate comparisons
6. **Interactive Dashboard** - Full exploratory data analysis tool

### 🖼️ **Sample Visualizations**

#### PnL Distribution Analysis

![PnL Distribution](reports/pnl_distribution_by_sentiment.png)
_4-panel visualization showing profit/loss patterns across sentiment phases_

#### Correlation Matrix

![Correlation Heatmap](reports/performance_heatmap.png)
_Statistical relationships between trading metrics and market sentiment_

#### Time Series Performance

![Time Series](reports/time_series_analysis.png)
_Temporal analysis of PnL, sentiment, and win rates over time_

## 🛠️ Technical Details

### **Dependencies**

```
pandas>=1.3.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
plotly>=4.14.0
scipy>=1.7.0
statsmodels>=0.12.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

---

## 🏁 Conclusion

This project provides **statistically significant evidence** that Bitcoin market sentiment impacts trader performance. The analysis offers a data-driven foundation for developing sentiment-aware trading strategies with clear statistical backing.

**Bottom Line**: The 55.5% PnL improvement during Greed periods, combined with higher win rates, presents a compelling case for integrating sentiment analysis into trading workflows.

---

_Analysis completed: February 19, 2025_  
_Dataset: 184,263 trades, $880M+ volume_  
_Statistical significance: p < 0.000001_

🔗 **Interactive Dashboard**: [Explore the Data](reports/interactive_dashboard.html)
