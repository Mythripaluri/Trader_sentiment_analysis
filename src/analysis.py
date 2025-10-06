"""
Statistical analysis utilities for trader sentiment analysis.
This module contains functions for hypothesis testing, correlation analysis, and modeling.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

def descriptive_statistics_by_sentiment(df):
    """
    Calculate descriptive statistics grouped by sentiment.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: Dictionary containing statistical summaries
    """
    print("=== Descriptive Statistics by Sentiment ===\n")
    
    results = {}
    
    # Overall statistics
    overall_stats = {
        'total_trades': len(df),
        'unique_accounts': df['account'].nunique(),
        'date_range': (df['merge_date'].min(), df['merge_date'].max()),
        'total_volume': df['size_usd'].sum(),
        'overall_win_rate': df['is_profitable'].mean()
    }
    
    results['overall'] = overall_stats
    
    # Statistics by sentiment
    sentiment_stats = df.groupby('sentiment_binary').agg({
        'closed_pnl': ['count', 'mean', 'std', 'median', 'sum'],
        'size_usd': ['mean', 'sum'],
        'is_profitable': 'mean',
        'leverage_approx': lambda x: x.dropna().mean(),
        'abs_pnl': 'mean'
    }).round(4)
    
    results['by_sentiment'] = sentiment_stats
    
    # Print results
    print(f"Overall Statistics:")
    print(f"- Total Trades: {overall_stats['total_trades']:,}")
    print(f"- Unique Accounts: {overall_stats['unique_accounts']}")
    print(f"- Date Range: {overall_stats['date_range'][0]} to {overall_stats['date_range'][1]}")
    print(f"- Total Volume: ${overall_stats['total_volume']:,.2f}")
    print(f"- Overall Win Rate: {overall_stats['overall_win_rate']:.1%}")
    
    print(f"\nStatistics by Sentiment:")
    print(sentiment_stats)
    
    return results

def test_pnl_differences_by_sentiment(df, alpha=0.05):
    """
    Test if there are significant differences in PnL between sentiment groups.
    
    Args:
        df (pd.DataFrame): Merged dataset
        alpha (float): Significance level
    
    Returns:
        dict: Test results
    """
    print("\n=== Testing PnL Differences by Sentiment ===\n")
    
    results = {}
    
    # Get PnL data for each sentiment group
    fear_pnl = df[df['sentiment_binary'] == 'Fear']['closed_pnl']
    greed_pnl = df[df['sentiment_binary'] == 'Greed']['closed_pnl']
    neutral_pnl = df[df['sentiment_binary'] == 'Neutral']['closed_pnl']
    
    # Test 1: Fear vs Greed (t-test)
    fear_greed_ttest = ttest_ind(fear_pnl, greed_pnl, equal_var=False)
    
    # Test 2: Fear vs Greed (Mann-Whitney U test - non-parametric)
    fear_greed_mw = mannwhitneyu(fear_pnl, greed_pnl, alternative='two-sided')
    
    # Test 3: All groups (ANOVA)
    all_groups = [fear_pnl, greed_pnl, neutral_pnl]
    anova_result = stats.f_oneway(*all_groups)
    
    results = {
        'fear_vs_greed_ttest': {
            'statistic': fear_greed_ttest.statistic,
            'p_value': fear_greed_ttest.pvalue,
            'significant': fear_greed_ttest.pvalue < alpha,
            'interpretation': 'Significant difference' if fear_greed_ttest.pvalue < alpha else 'No significant difference'
        },
        'fear_vs_greed_mannwhitney': {
            'statistic': fear_greed_mw.statistic,
            'p_value': fear_greed_mw.pvalue,
            'significant': fear_greed_mw.pvalue < alpha,
            'interpretation': 'Significant difference' if fear_greed_mw.pvalue < alpha else 'No significant difference'
        },
        'anova_all_groups': {
            'f_statistic': anova_result.statistic,
            'p_value': anova_result.pvalue,
            'significant': anova_result.pvalue < alpha,
            'interpretation': 'Significant differences exist' if anova_result.pvalue < alpha else 'No significant differences'
        }
    }
    
    # Print results
    print(f"1. Fear vs Greed (t-test):")
    print(f"   - Mean PnL Fear: ${fear_pnl.mean():.2f}")
    print(f"   - Mean PnL Greed: ${greed_pnl.mean():.2f}")
    print(f"   - t-statistic: {fear_greed_ttest.statistic:.4f}")
    print(f"   - p-value: {fear_greed_ttest.pvalue:.6f}")
    print(f"   - Result: {results['fear_vs_greed_ttest']['interpretation']}")
    
    print(f"\n2. Fear vs Greed (Mann-Whitney U test):")
    print(f"   - U-statistic: {fear_greed_mw.statistic:.0f}")
    print(f"   - p-value: {fear_greed_mw.pvalue:.6f}")
    print(f"   - Result: {results['fear_vs_greed_mannwhitney']['interpretation']}")
    
    print(f"\n3. All Groups ANOVA:")
    print(f"   - F-statistic: {anova_result.statistic:.4f}")
    print(f"   - p-value: {anova_result.pvalue:.6f}")
    print(f"   - Result: {results['anova_all_groups']['interpretation']}")
    
    return results

def test_win_rate_differences(df, alpha=0.05):
    """
    Test if win rates differ significantly between sentiment groups.
    
    Args:
        df (pd.DataFrame): Merged dataset
        alpha (float): Significance level
    
    Returns:
        dict: Test results
    """
    print("\n=== Testing Win Rate Differences by Sentiment ===\n")
    
    # Calculate win rates and counts
    sentiment_summary = df.groupby('sentiment_binary').agg({
        'is_profitable': ['sum', 'count', 'mean']
    }).round(4)
    
    sentiment_summary.columns = ['wins', 'total', 'win_rate']
    
    print("Win Rate Summary by Sentiment:")
    print(sentiment_summary)
    
    # Prepare data for proportions test
    fear_data = sentiment_summary.loc['Fear']
    greed_data = sentiment_summary.loc['Greed']
    
    # Two-proportion z-test
    successes = np.array([fear_data['wins'], greed_data['wins']])
    nobs = np.array([fear_data['total'], greed_data['total']])
    
    z_stat, p_value = proportions_ztest(successes, nobs)
    
    # Chi-square test for all groups
    contingency_table = pd.crosstab(df['sentiment_binary'], df['is_profitable'])
    chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
    
    results = {
        'fear_vs_greed_proportions': {
            'fear_win_rate': fear_data['win_rate'],
            'greed_win_rate': greed_data['win_rate'],
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'interpretation': 'Significant difference' if p_value < alpha else 'No significant difference'
        },
        'chi_square_all_groups': {
            'chi2_statistic': chi2,
            'p_value': p_chi2,
            'degrees_of_freedom': dof,
            'significant': p_chi2 < alpha,
            'interpretation': 'Significant association' if p_chi2 < alpha else 'No significant association'
        },
        'contingency_table': contingency_table
    }
    
    print(f"\nFear vs Greed Win Rate Test:")
    print(f"- Fear Win Rate: {fear_data['win_rate']:.1%}")
    print(f"- Greed Win Rate: {greed_data['win_rate']:.1%}")
    print(f"- Z-statistic: {z_stat:.4f}")
    print(f"- P-value: {p_value:.6f}")
    print(f"- Result: {results['fear_vs_greed_proportions']['interpretation']}")
    
    print(f"\nChi-square Test (All Groups):")
    print(f"- Chi-square statistic: {chi2:.4f}")
    print(f"- P-value: {p_chi2:.6f}")
    print(f"- Result: {results['chi_square_all_groups']['interpretation']}")
    
    return results

def correlation_analysis(df):
    """
    Analyze correlations between sentiment and trading metrics.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: Correlation results
    """
    print("\n=== Correlation Analysis ===\n")
    
    # Select relevant columns for correlation
    correlation_cols = [
        'closed_pnl', 'size_usd', 'value', 'sentiment_score_normalized',
        'leverage_approx', 'is_profitable', 'abs_pnl'
    ]
    
    # Calculate correlation matrix
    corr_matrix = df[correlation_cols].corr()
    
    # Focus on sentiment correlations
    sentiment_corr = corr_matrix['sentiment_score_normalized'].sort_values(key=abs, ascending=False)
    
    print("Correlation with Sentiment Score:")
    for metric, corr in sentiment_corr.items():
        if metric != 'sentiment_score_normalized':
            strength = 'Strong' if abs(corr) > 0.5 else ('Moderate' if abs(corr) > 0.3 else 'Weak')
            direction = 'Positive' if corr > 0 else 'Negative'
            print(f"- {metric}: {corr:.4f} ({strength} {direction})")
    
    # Test significance of key correlations
    from scipy.stats import pearsonr
    
    key_correlations = {}
    for col in ['closed_pnl', 'is_profitable', 'leverage_approx', 'size_usd']:
        if col in df.columns:
            # Remove NaN values for correlation test
            valid_data = df[['sentiment_score_normalized', col]].dropna()
            if len(valid_data) > 0:
                corr_coef, p_value = pearsonr(valid_data['sentiment_score_normalized'], valid_data[col])
                key_correlations[col] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'n_observations': len(valid_data)
                }
    
    print(f"\nSignificance Tests for Key Correlations:")
    for metric, stats in key_correlations.items():
        sig_text = "Significant" if stats['significant'] else "Not significant"
        print(f"- {metric}: r={stats['correlation']:.4f}, p={stats['p_value']:.6f} ({sig_text})")
    
    return {
        'correlation_matrix': corr_matrix,
        'sentiment_correlations': sentiment_corr,
        'significance_tests': key_correlations
    }

def sentiment_impact_analysis(df):
    """
    Analyze the impact of different sentiment levels on trading performance.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: Impact analysis results
    """
    print("\n=== Sentiment Impact Analysis ===\n")
    
    # Create sentiment intensity categories
    df['sentiment_intensity'] = pd.cut(df['value'], 
                                      bins=[0, 20, 40, 60, 80, 100],
                                      labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'])
    
    # Analyze performance by sentiment intensity
    intensity_analysis = df.groupby('sentiment_intensity').agg({
        'closed_pnl': ['count', 'mean', 'std', 'median'],
        'is_profitable': 'mean',
        'size_usd': 'mean',
        'leverage_approx': lambda x: x.dropna().mean()
    }).round(4)
    
    intensity_analysis.columns = ['_'.join(col).strip() for col in intensity_analysis.columns.values]
    
    print("Performance by Sentiment Intensity:")
    print(intensity_analysis)
    
    # Calculate effect sizes (Cohen's d) between extreme groups
    extreme_fear = df[df['sentiment_intensity'] == 'Extreme Fear']['closed_pnl']
    extreme_greed = df[df['sentiment_intensity'] == 'Extreme Greed']['closed_pnl']
    
    if len(extreme_fear) > 0 and len(extreme_greed) > 0:
        pooled_std = np.sqrt(((len(extreme_fear) - 1) * extreme_fear.std()**2 + 
                             (len(extreme_greed) - 1) * extreme_greed.std()**2) / 
                            (len(extreme_fear) + len(extreme_greed) - 2))
        
        cohens_d = (extreme_greed.mean() - extreme_fear.mean()) / pooled_std
        
        effect_size_interpretation = ('Large' if abs(cohens_d) >= 0.8 else
                                    ('Medium' if abs(cohens_d) >= 0.5 else
                                     ('Small' if abs(cohens_d) >= 0.2 else 'Negligible')))
        
        print(f"\nEffect Size Analysis (Extreme Fear vs Extreme Greed):")
        print(f"- Cohen's d: {cohens_d:.4f}")
        print(f"- Interpretation: {effect_size_interpretation} effect")
        print(f"- Extreme Fear mean PnL: ${extreme_fear.mean():.2f}")
        print(f"- Extreme Greed mean PnL: ${extreme_greed.mean():.2f}")
    
    return {
        'intensity_analysis': intensity_analysis,
        'cohens_d': cohens_d if 'cohens_d' in locals() else None,
        'effect_size_interpretation': effect_size_interpretation if 'effect_size_interpretation' in locals() else None
    }

def trading_behavior_analysis(df):
    """
    Analyze how trading behavior changes with market sentiment.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: Behavior analysis results
    """
    print("\n=== Trading Behavior Analysis ===\n")
    
    behavior_metrics = df.groupby('sentiment_binary').agg({
        'size_usd': ['mean', 'median', 'std'],
        'leverage_approx': lambda x: x.dropna().mean(),
        'is_long': 'mean',  # Proportion of long positions
        'hour': 'mean',  # Average trading hour
        'day_of_week': 'mean'  # Average day of week
    }).round(4)
    
    print("Trading Behavior by Sentiment:")
    print(behavior_metrics)
    
    # Test if position direction is associated with sentiment
    direction_contingency = pd.crosstab(df['sentiment_binary'], df['is_long'])
    chi2_direction, p_direction, _, _ = chi2_contingency(direction_contingency)
    
    print(f"\nPosition Direction vs Sentiment:")
    print(direction_contingency)
    print(f"Chi-square test: χ² = {chi2_direction:.4f}, p = {p_direction:.6f}")
    print(f"Result: {'Significant association' if p_direction < 0.05 else 'No significant association'}")
    
    return {
        'behavior_metrics': behavior_metrics,
        'direction_association': {
            'contingency_table': direction_contingency,
            'chi2_statistic': chi2_direction,
            'p_value': p_direction,
            'significant': p_direction < 0.05
        }
    }

def run_comprehensive_analysis(df):
    """
    Run all statistical analyses and return comprehensive results.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: All analysis results
    """
    print("🔍 Running Comprehensive Statistical Analysis\n")
    print("=" * 60)
    
    results = {}
    
    # Run all analyses
    results['descriptive_stats'] = descriptive_statistics_by_sentiment(df)
    results['pnl_tests'] = test_pnl_differences_by_sentiment(df)
    results['win_rate_tests'] = test_win_rate_differences(df)
    results['correlations'] = correlation_analysis(df)
    results['sentiment_impact'] = sentiment_impact_analysis(df)
    results['behavior_analysis'] = trading_behavior_analysis(df)
    
    print("\n" + "=" * 60)
    print("✅ Comprehensive Statistical Analysis Completed")
    
    return results

# Optional: Simple predictive modeling
def simple_predictive_model(df):
    """
    Build a simple model to predict sentiment based on trading behavior.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        dict: Model results
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        
        print("\n=== Simple Predictive Modeling ===\n")
        
        # Prepare features (trading behavior) and target (sentiment)
        feature_cols = ['closed_pnl', 'size_usd', 'leverage_approx', 'is_long', 'hour', 'day_of_week']
        
        # Remove rows with missing values
        model_data = df[feature_cols + ['sentiment_binary']].dropna()
        
        X = model_data[feature_cols]
        y = model_data['sentiment_binary']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Train model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"- Accuracy: {accuracy:.4f}")
        print(f"- Training samples: {len(X_train):,}")
        print(f"- Test samples: {len(X_test):,}")
        
        print(f"\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"- {row['feature']}: {row['importance']:.4f}")
        
        return {
            'model': rf_model,
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'classification_report': classification_report(y_test, y_pred)
        }
        
    except ImportError:
        print("Scikit-learn not available for predictive modeling")
        return None
    except Exception as e:
        print(f"Error in predictive modeling: {e}")
        return None