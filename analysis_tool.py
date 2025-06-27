import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import json
from datetime import datetime

# Column mapping dictionary - maps possible column names to standard names
COLUMN_MAPPING = {
    # Timestamp
    'timestamp': ['timestamp', 'date', 'time', 'datetime'],
    
    # Department
    'department': ['department', 'dept', 'division', 'team'],
    
    # Job Role
    'job_role': ['job_role', 'role', 'position', 'job', 'job role'],
    
    # AI Tool
    'ai_tool': ['ai_tool_used', 'ai tool', 'tool', 'ai', 'ai tool used'],
    
    # Usage Frequency
    'usage_frequency': ['usage_frequency', 'frequency', 'usage', 'how often'],
    
    # Purpose
    'purpose': ['purpose', 'use case', 'application', 'used for'],
    
    # Ease of Use
    'ease_of_use': ['ease_of_use', 'ease', 'usability', 'ease of use'],
    
    # Time Saved
    'time_saved': ['time_saved', 'time', 'efficiency', 'time save', 'time saving'],
    
    # Suggestions
    'suggestions': ['improvement_suggestion', 'suggestions', 'feedback', 'comments']
}

def standardize_columns(df):
    """Standardize column names using flexible mapping"""
    # Create reverse mapping (variation -> standard name)
    reverse_mapping = {}
    for standard_name, variations in COLUMN_MAPPING.items():
        for variation in variations:
            reverse_mapping[variation.lower().replace(' ', '_')] = standard_name
    
    # Normalize existing columns
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    
    # Map to standard names
    df.columns = [reverse_mapping.get(col, col) for col in df.columns]
    
    return df

def load_data(uploaded_file):
    """Load and standardize data from uploaded file"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df = pd.read_json(uploaded_file)
    else:
        st.error("Unsupported file format")
        return None
    
    df = standardize_columns(df)
    return df

def plot_usage_frequency(df):
    """Plot usage frequency with flexible data handling"""
    if 'usage_frequency' not in df.columns:
        st.warning("No usage frequency data available")
        return None
    
    # Normalize frequency values
    freq_map = {
        'daily': 'Daily',
        'weekly': 'Weekly',
        'monthly': 'Monthly',
        'rarely': 'Rarely'
    }
    
    df['usage_frequency'] = df['usage_frequency'].str.lower().map(freq_map).fillna('Other')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x='usage_frequency',
        order=['Daily', 'Weekly', 'Monthly', 'Rarely', 'Other'],
        ax=ax
    )
    ax.set_title('AI Tools Usage Frequency Distribution')
    ax.set_xlabel('Usage Frequency')
    ax.set_ylabel('Count')
    plt.tight_layout()
    return fig

def plot_tool_popularity(df):
    """Plot AI tool popularity with flexible data handling"""
    if 'ai_tool' not in df.columns:
        st.warning("No AI tool data available")
        return None
    
    # Clean tool names
    df['ai_tool'] = df['ai_tool'].str.strip().str.title()
    
    fig = px.pie(
        df,
        names='ai_tool',
        title='Popularity of AI Tools Among Staff'
    )
    return fig

def plot_metrics(df):
    """Plot ease of use and time saved metrics"""
    figs = []
    
    if 'ease_of_use' in df.columns:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['ease_of_use'], bins=5, kde=True, ax=ax1)
        ax1.set_title('Ease of Use Ratings')
        ax1.set_xlabel('Rating (1-5)')
        figs.append(fig1)
    
    if 'time_saved' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['time_saved'], bins=5, kde=True, ax=ax2)
        ax2.set_title('Time Saved Ratings')
        ax2.set_xlabel('Rating (1-5)')
        figs.append(fig2)
    
    return figs if figs else None

def create_dashboard():
    st.set_page_config(page_title="AI Tools Usage Analysis", layout="wide")
    st.title("AI Tools Usage Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload your data (CSV or JSON)", type=['csv', 'json'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            st.subheader("Usage Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                freq_fig = plot_usage_frequency(df)
                if freq_fig:
                    st.pyplot(freq_fig)
            
            with col2:
                tool_fig = plot_tool_popularity(df)
                if tool_fig:
                    st.plotly_chart(tool_fig)
            
            st.subheader("User Experience Metrics")
            metric_figs = plot_metrics(df)
            if metric_figs:
                cols = st.columns(len(metric_figs))
                for i, fig in enumerate(metric_figs):
                    with cols[i]:
                        st.pyplot(fig)

if __name__ == '__main__':
    create_dashboard()