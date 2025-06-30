import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import re
from collections import Counter

# Enhanced column mapping
COLUMN_MAPPING = {
    'timestamp': ['timestamp', 'date', 'time', 'datetime'],
    'department': ['department', 'dept', 'division', 'team', 'which department'],
    'job_role': ['job_role', 'role', 'position', 'job', 'job role', 'role:'],
    'ai_tool': ['ai_tool', 'what ai tool', 'ai tool used', 'tool', 'ai'],
    'usage_frequency': ['usage_frequency', 'usage of ai tools', 'frequency', 'usage', 'how often'],
    'purpose': ['purpose', 'purpose of using ai tools', 'use case', 'application', 'used for'],
    'ease_of_use': ['ease_of_use', 'ease', 'usability', 'ease of use'],
    'efficiency': ['efficiency', 'how efficiency', 'time_saved', 'time save', 'time saving'],
    'suggestions': ['suggestions', 'improvement', 'feedback', 'comments', 'any suggestions']
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """More robust column standardization"""
    column_mapping = {}
    
    for original_col in df.columns:
        original_lower = str(original_col).lower().strip()
        matched = False
        
        for std_col, variations in COLUMN_MAPPING.items():
            for variation in variations:
                if variation.lower() in original_lower:
                    column_mapping[original_col] = std_col
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            column_mapping[original_col] = original_col
    
    # Apply mapping and drop duplicates
    df = df.rename(columns=column_mapping)
    return df.loc[:, ~df.columns.duplicated()]

def load_data(uploaded_file) -> pd.DataFrame:
    """Robust data loader with enhanced column handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON.")
            return None
        
        if df.empty:
            st.error("Uploaded file is empty")
            return None
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Store debug info
        st.session_state['processed_columns'] = df.columns.tolist()
        st.session_state['processed_data_sample'] = df.head(3).to_dict('records')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def plot_usage_frequency(df: pd.DataFrame):
    """Enhanced usage frequency visualization"""
    if 'usage_frequency' not in df.columns:
        st.error("Usage frequency column not found in data")
        return None
    
    # Clean frequency data
    freq_map = {
        r'daily': 'Daily',
        r'weekly': 'Weekly',
        r'monthly': 'Monthly',
        r'rarely': 'Rarely',
        r'never': 'Never',
        r'^no': 'Never',
        r'^yes': 'Regularly'
    }
    
    usage_data = df['usage_frequency'].astype(str).str.lower().str.strip()
    for pattern, replacement in freq_map.items():
        usage_data = usage_data.str.replace(pattern, replacement, regex=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
    
    counts = usage_data.value_counts().reindex(order, fill_value=0)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis', order=order)
    
    ax.set_title('AI Tools Usage Frequency')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    return fig

def plot_tool_popularity(df: pd.DataFrame):
    """Enhanced tool popularity visualization"""
    if 'ai_tool' not in df.columns:
        st.error("AI tool column not found in data")
        return None
    
    # Clean tool names
    tool_map = {
        r'chat.?gpt': 'ChatGPT',
        r'poe': 'Poe',
        r'canva': 'Canva',
        r'gamma': 'Gamma',
        r'mid.?journey': 'Midjourney',
        r'copilot': 'Copilot',
        r'kling.?ai': 'Kling AI',
        r'deep.?seek': 'Deepseek'
    }
    
    tool_data = df['ai_tool'].astype(str).str.strip().str.lower()
    for pattern, replacement in tool_map.items():
        tool_data = tool_data.str.replace(pattern, replacement, regex=True)
    
    # Count and plot
    tool_counts = tool_data.value_counts().reset_index()
    tool_counts.columns = ['AI Tool', 'Count']
    
    fig = px.bar(
        tool_counts,
        x='AI Tool',
        y='Count',
        title='AI Tool Popularity',
        color='AI Tool',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=600
    )
    return fig

def create_dashboard():
    st.set_page_config(
        page_title="AI Tools Usage Analysis",
        layout="wide",
        page_icon="🤖"
    )
    
    st.title("📊 AI Tools Usage Analysis Dashboard")
    st.markdown("Analyze AI tool usage within your organization")
    
    uploaded_file = st.file_uploader(
        "Upload your data (CSV, Excel, or JSON)",
        type=['csv', 'xlsx', 'xls', 'json']
    )
    
    if uploaded_file is not None:
        with st.spinner('Analyzing data...'):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            with st.expander("🔍 Data Debug Info"):
                st.write("### Processed Columns:")
                st.write(st.session_state.get('processed_columns', []))
                
                st.write("### Data Sample:")
                st.write(st.session_state.get('processed_data_sample', {}))
            
            st.subheader("📋 Data Overview")
            st.dataframe(df.head())
            
            st.subheader("📈 Usage Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                freq_fig = plot_usage_frequency(df)
                if freq_fig:
                    st.pyplot(freq_fig)
            
            with col2:
                tool_fig = plot_tool_popularity(df)
                if tool_fig:
                    st.plotly_chart(tool_fig)
            
            st.subheader("💾 Download Processed Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="ai_usage_analysis.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    create_dashboard()
