"""
Visualization utilities for trader sentiment analysis.
This module contains functions for creating various charts and plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

def setup_plot_style():
    """Setup consistent plot styling"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    sns.set_style("whitegrid")

def plot_pnl_distribution_by_sentiment(df, figsize=(15, 10)):
    """
    Plot PnL distribution by sentiment with multiple visualizations.
    
    Args:
        df (pd.DataFrame): Merged dataset
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('PnL Distribution by Market Sentiment', fontsize=16, fontweight='bold')
    
    # Remove extreme outliers for better visualization
    pnl_q99 = df['closed_pnl'].quantile(0.99)
    pnl_q01 = df['closed_pnl'].quantile(0.01)
    df_clean = df[(df['closed_pnl'] >= pnl_q01) & (df['closed_pnl'] <= pnl_q99)].copy()
    
    # 1. Box plot
    sns.boxplot(data=df_clean, x='sentiment_binary', y='closed_pnl', ax=axes[0,0])
    axes[0,0].set_title('PnL Distribution by Sentiment (Box Plot)')
    axes[0,0].set_ylabel('Closed PnL ($)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Violin plot
    sns.violinplot(data=df_clean, x='sentiment_binary', y='closed_pnl', ax=axes[0,1])
    axes[0,1].set_title('PnL Distribution by Sentiment (Violin Plot)')
    axes[0,1].set_ylabel('Closed PnL ($)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Histogram with overlaid densities
    for sentiment in df_clean['sentiment_binary'].unique():
        data = df_clean[df_clean['sentiment_binary'] == sentiment]['closed_pnl']
        axes[1,0].hist(data, alpha=0.6, label=sentiment, bins=50, density=True)
    axes[1,0].set_title('PnL Density Distribution by Sentiment')
    axes[1,0].set_xlabel('Closed PnL ($)')
    axes[1,0].set_ylabel('Density')
    axes[1,0].legend()
    
    # 4. Mean PnL by sentiment
    mean_pnl = df.groupby('sentiment_binary')['closed_pnl'].mean().sort_values(ascending=False)
    colors = ['green' if x > 0 else 'red' for x in mean_pnl.values]
    bars = axes[1,1].bar(mean_pnl.index, mean_pnl.values, color=colors, alpha=0.7)
    axes[1,1].set_title('Average PnL by Sentiment')
    axes[1,1].set_ylabel('Average Closed PnL ($)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_pnl.values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 
                      (0.05 * bar.get_height() if value > 0 else -0.15 * abs(bar.get_height())), 
                      f'${value:.2f}', ha='center', va='bottom' if value > 0 else 'top')
    
    plt.tight_layout()
    return fig

def plot_leverage_analysis(df, figsize=(15, 8)):
    """
    Analyze leverage usage patterns by sentiment.
    
    Args:
        df (pd.DataFrame): Merged dataset
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    setup_plot_style()
    
    # Filter out extreme leverage values and NaN
    df_lev = df.dropna(subset=['leverage_approx'])
    df_lev = df_lev[(df_lev['leverage_approx'] > 0) & (df_lev['leverage_approx'] <= 50)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Leverage Analysis by Market Sentiment', fontsize=16, fontweight='bold')
    
    # 1. Leverage distribution by sentiment
    sns.boxplot(data=df_lev, x='sentiment_binary', y='leverage_approx', ax=axes[0])
    axes[0].set_title('Leverage Distribution by Sentiment')
    axes[0].set_ylabel('Approximate Leverage')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Average leverage by sentiment
    avg_leverage = df_lev.groupby('sentiment_binary')['leverage_approx'].mean().sort_values(ascending=False)
    bars = axes[1].bar(avg_leverage.index, avg_leverage.values, alpha=0.7)
    axes[1].set_title('Average Leverage by Sentiment')
    axes[1].set_ylabel('Average Leverage')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, avg_leverage.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05 * bar.get_height(), 
                    f'{value:.2f}x', ha='center', va='bottom')
    
    # 3. Leverage vs PnL scatter plot
    sentiment_colors = {'Fear': 'red', 'Greed': 'green', 'Neutral': 'blue'}
    for sentiment in df_lev['sentiment_binary'].unique():
        data = df_lev[df_lev['sentiment_binary'] == sentiment]
        axes[2].scatter(data['leverage_approx'], data['closed_pnl'], 
                       alpha=0.3, label=sentiment, color=sentiment_colors.get(sentiment, 'gray'))
    
    axes[2].set_title('Leverage vs PnL by Sentiment')
    axes[2].set_xlabel('Approximate Leverage')
    axes[2].set_ylabel('Closed PnL ($)')
    axes[2].legend()
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_time_series_analysis(df, figsize=(15, 10)):
    """
    Create time series analysis of PnL and sentiment.
    
    Args:
        df (pd.DataFrame): Merged dataset
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    setup_plot_style()
    
    # Aggregate data by date
    daily_data = df.groupby('merge_date').agg({
        'closed_pnl': 'mean',
        'value': 'first',  # Sentiment value is the same for all trades on the same day
        'sentiment_binary': 'first',
        'is_profitable': 'mean',
        'size_usd': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle('Time Series Analysis: PnL vs Market Sentiment', fontsize=16, fontweight='bold')
    
    # 1. Average daily PnL
    axes[0].plot(daily_data['merge_date'], daily_data['closed_pnl'], 
                color='blue', linewidth=1.5, alpha=0.8)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Daily Average PnL')
    axes[0].set_ylabel('Average PnL ($)')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Sentiment value over time
    sentiment_colors = daily_data['value'].apply(lambda x: 'red' if x <= 40 else ('green' if x >= 60 else 'yellow'))
    axes[1].scatter(daily_data['merge_date'], daily_data['value'], 
                   c=sentiment_colors, alpha=0.7, s=20)
    axes[1].axhline(y=50, color='black', linestyle='-', alpha=0.5, label='Neutral (50)')
    axes[1].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Extreme Fear (20)')
    axes[1].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Extreme Greed (80)')
    axes[1].set_title('Bitcoin Fear & Greed Index Over Time')
    axes[1].set_ylabel('Sentiment Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Win rate over time
    axes[2].plot(daily_data['merge_date'], daily_data['is_profitable'] * 100, 
                color='orange', linewidth=1.5, alpha=0.8)
    axes[2].axhline(y=50, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Daily Win Rate')
    axes[2].set_ylabel('Win Rate (%)')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_performance_heatmap(df, figsize=(12, 8)):
    """
    Create correlation heatmap of key metrics.
    
    Args:
        df (pd.DataFrame): Merged dataset
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    setup_plot_style()
    
    # Select numeric columns for correlation
    numeric_cols = ['closed_pnl', 'size_usd', 'value', 'sentiment_score_normalized', 
                   'leverage_approx', 'is_profitable', 'abs_pnl', 'hour', 'day_of_week']
    
    # Calculate correlation matrix
    correlation_data = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
    
    ax.set_title('Correlation Heatmap: Trading Metrics vs Sentiment', 
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_win_rate_analysis(df, figsize=(15, 8)):
    """
    Analyze win rates across different sentiment conditions.
    
    Args:
        df (pd.DataFrame): Merged dataset
        figsize (tuple): Figure size
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Win Rate Analysis by Market Sentiment', fontsize=16, fontweight='bold')
    
    # 1. Overall win rate by sentiment
    win_rates = df.groupby('sentiment_binary')['is_profitable'].mean() * 100
    colors = ['red', 'green', 'orange']
    bars = axes[0].bar(win_rates.index, win_rates.values, color=colors, alpha=0.7)
    axes[0].axhline(y=50, color='black', linestyle='--', alpha=0.5)
    axes[0].set_title('Win Rate by Sentiment')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, win_rates.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
    
    # 2. Win rate by sentiment category (more granular)
    win_rates_detailed = df.groupby('sentiment_category')['is_profitable'].mean() * 100
    win_rates_detailed = win_rates_detailed.sort_values(ascending=False)
    
    bars = axes[1].bar(range(len(win_rates_detailed)), win_rates_detailed.values, 
                      alpha=0.7, color='lightblue')
    axes[1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Win Rate by Detailed Sentiment')
    axes[1].set_ylabel('Win Rate (%)')
    axes[1].set_xticks(range(len(win_rates_detailed)))
    axes[1].set_xticklabels(win_rates_detailed.index, rotation=45, ha='right')
    
    # Add value labels
    for i, value in enumerate(win_rates_detailed.values):
        axes[1].text(i, value + 1, f'{value:.1f}%', ha='center', va='bottom')
    
    # 3. Win rate vs sentiment value (continuous)
    # Bin sentiment values for better visualization
    df['sentiment_bins'] = pd.cut(df['value'], bins=10, labels=False)
    bin_win_rates = df.groupby('sentiment_bins')['is_profitable'].mean() * 100
    bin_centers = df.groupby('sentiment_bins')['value'].mean()
    
    axes[2].plot(bin_centers, bin_win_rates, marker='o', linewidth=2, markersize=6)
    axes[2].axhline(y=50, color='black', linestyle='--', alpha=0.5)
    axes[2].axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('Win Rate vs Sentiment Value')
    axes[2].set_xlabel('Sentiment Value (Fear ← → Greed)')
    axes[2].set_ylabel('Win Rate (%)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_interactive_dashboard(df):
    """
    Create an interactive Plotly dashboard.
    
    Args:
        df (pd.DataFrame): Merged dataset
    
    Returns:
        plotly.graph_objects.Figure: Interactive dashboard
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PnL by Sentiment', 'Trading Volume by Sentiment', 
                       'Win Rate by Sentiment', 'Sentiment vs PnL Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    # 1. PnL by sentiment (box plot)
    for sentiment in df['sentiment_binary'].unique():
        data = df[df['sentiment_binary'] == sentiment]['closed_pnl']
        fig.add_trace(
            go.Box(y=data, name=sentiment, boxpoints='outliers'),
            row=1, col=1
        )
    
    # 2. Trading volume by sentiment
    volume_by_sentiment = df.groupby('sentiment_binary')['size_usd'].sum().reset_index()
    fig.add_trace(
        go.Bar(x=volume_by_sentiment['sentiment_binary'], 
               y=volume_by_sentiment['size_usd'],
               name='Trading Volume'),
        row=1, col=2
    )
    
    # 3. Win rate by sentiment
    win_rate_by_sentiment = df.groupby('sentiment_binary')['is_profitable'].mean().reset_index()
    win_rate_by_sentiment['is_profitable'] *= 100
    
    fig.add_trace(
        go.Bar(x=win_rate_by_sentiment['sentiment_binary'], 
               y=win_rate_by_sentiment['is_profitable'],
               name='Win Rate (%)', marker_color='lightblue'),
        row=2, col=1
    )
    
    # 4. Time series: sentiment and average PnL
    daily_data = df.groupby('merge_date').agg({
        'closed_pnl': 'mean',
        'value': 'first'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(x=daily_data['merge_date'], y=daily_data['value'],
                  mode='lines', name='Sentiment Value', line=dict(color='red')),
        row=2, col=2, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=daily_data['merge_date'], y=daily_data['closed_pnl'],
                  mode='lines', name='Avg PnL', line=dict(color='blue')),
        row=2, col=2, secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title_text="Trader Performance vs Market Sentiment Dashboard",
        showlegend=True,
        height=800
    )
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Closed PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Trading Volume ($)", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Value", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Average PnL ($)", row=2, col=2, secondary_y=True)
    
    return fig

def save_all_plots(df, output_dir="reports/"):
    """
    Generate and save all plots to files.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualization plots...")
    
    # Generate all plots
    fig1 = plot_pnl_distribution_by_sentiment(df)
    fig1.savefig(f"{output_dir}pnl_distribution_by_sentiment.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = plot_leverage_analysis(df)
    fig2.savefig(f"{output_dir}leverage_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = plot_time_series_analysis(df)
    fig3.savefig(f"{output_dir}time_series_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    fig4 = plot_performance_heatmap(df)
    fig4.savefig(f"{output_dir}performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    fig5 = plot_win_rate_analysis(df)
    fig5.savefig(f"{output_dir}win_rate_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # Save interactive dashboard
    fig6 = create_interactive_dashboard(df)
    fig6.write_html(f"{output_dir}interactive_dashboard.html")
    
    print(f"All plots saved to {output_dir}")
    
    return {
        'pnl_distribution': f"{output_dir}pnl_distribution_by_sentiment.png",
        'leverage_analysis': f"{output_dir}leverage_analysis.png",
        'time_series': f"{output_dir}time_series_analysis.png",
        'heatmap': f"{output_dir}performance_heatmap.png",
        'win_rate': f"{output_dir}win_rate_analysis.png",
        'dashboard': f"{output_dir}interactive_dashboard.html"
    }