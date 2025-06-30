import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import re
from collections import Counter
from wordcloud import WordCloud

# Ultra-comprehensive column mapping
COLUMN_MAPPING = {
    'timestamp': ['timestamp', 'date', 'time', 'datetime', 'time stamp', 'recorded at'],
    'department': ['department', 'dept', 'division', 'team', 'group', 'business unit'],
    'job_role': ['job_role', 'role', 'position', 'job', 'job role', 'title', 'occupation', 'role:'],
    'ai_tool': ['ai_tool_used', 'ai tool', 'ai', 'ai tool used', 'which ai', 'ai system', 'application used', 'technology used'],
    'usage_frequency': ['usage_frequency', 'frequency', 'usage', 'how often', 'usage frequency', 'usage of tools', 'how frequently', 'rate of use', 'utilization'],
    'purpose': ['purpose', 'use case', 'application', 'used for', 'primary use', 'main purpose', 'how used', 'purpose of using ai tools'],
    'ease_of_use': ['ease_of_use', 'ease', 'usability', 'ease of use', 'user friendly', 'difficulty', 'how easy'],
    'time_saved': ['time_saved', 'time', 'efficiency', 'time save', 'time saving', 'productivity', 'hours saved'],
    'suggestions': ['improvement_suggestion', 'suggestions', 'feedback', 'comments', 'recommendations', 'ideas']
}

def debug_columns(df, context=""):
    """Helper function to display column debug info"""
    st.write(f"🔍 {context} - First 3 rows:")
    st.write(df.head(3))
    st.write("📋 All columns:", df.columns.tolist())

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bulletproof column standardization"""
    original_columns = df.columns.tolist()
    
    # Create case-insensitive mapping
    variation_map = {}
    for std_name, variations in COLUMN_MAPPING.items():
        for var in variations:
            norm_var = re.sub(r'[^a-z0-9]', '', var.lower())
            variation_map[norm_var] = std_name
    
    # Normalize all input columns
    new_columns = []
    seen_columns = set()
    
    for orig_col in df.columns:
        # Create normalized version for matching
        norm_col = re.sub(r'[^a-z0-9]', '', str(orig_col).lower())
        
        # Find best match (allowing partial matches)
        matched = False
        for pattern, std_col in variation_map.items():
            if re.search(pattern, norm_col):
                # Handle duplicates
                if std_col in seen_columns:
                    counter = 2
                    new_col = f"{std_col}_{counter}"
                    while new_col in seen_columns:
                        counter += 1
                        new_col = f"{std_col}_{counter}"
                else:
                    new_col = std_col
                
                new_columns.append(new_col)
                seen_columns.add(new_col)
                matched = True
                break
        
        if not matched:
            new_columns.append(orig_col)
    
    df.columns = new_columns
    st.session_state['column_mapping'] = dict(zip(original_columns, new_columns))
    return df

def load_data(uploaded_file) -> pd.DataFrame:
    """Robust data loader with detailed error handling"""
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
        
        # Initial debug info
        st.session_state['raw_data_sample'] = df.head(3).to_dict('records')
        st.session_state['raw_columns'] = df.columns.tolist()
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Post-standardization debug info
        st.session_state['processed_data_sample'] = df.head(3).to_dict('records')
        st.session_state['processed_columns'] = df.columns.tolist()
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def plot_usage_frequency(df: pd.DataFrame):
    """Foolproof usage frequency analysis"""
    # Try all possible column names
    possible_cols = [
        col for col in df.columns 
        if any(keyword in col.lower() 
              for keyword in ['usage', 'freq', 'often', 'utiliz'])
    ]
    
    if not possible_cols:
        st.error("""
        No usage frequency column found. I looked for columns containing:
        'usage', 'frequency', 'often', or 'utilization'
        
        Available columns:
        """ + str(df.columns.tolist()))
        return None
    
    usage_col = possible_cols[0]  # Use the first match
    
    # Enhanced cleaning
    freq_map = {
        r'daily|every day|each day': 'Daily',
        r'weekly|every week|each week': 'Weekly',
        r'monthly|every month|each month': 'Monthly',
        r'rarely|sometimes|occasionally|seldom': 'Rarely',
        r'never|not at all|haven\'t used': 'Never',
        r'often|regularly|frequently': 'Often',
        r'^1$|once': 'Once',
        r'^2$|twice': 'Twice',
        r'^3$|thrice': 'Thrice'
    }
    
    # Clean and categorize
    usage_data = df[usage_col].astype(str).str.lower().str.strip()
    categorized = usage_data.replace(freq_map, regex=True)
    
    # Handle numeric frequencies (1-7 times per week)
    if usage_data.str.isnumeric().any():
        num_data = pd.to_numeric(usage_data, errors='coerce')
        categorized = categorized.where(
            num_data.isna(),
            num_data.apply(lambda x: f"{int(x)} times" if not pd.isna(x) else 'Other')
        )
    
    # Final cleanup
    categorized = categorized.fillna('Unknown').replace('', 'Unknown')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ['Daily', 'Often', 'Weekly', 'Monthly', 'Rarely', 'Never', 
             'Once', 'Twice', 'Thrice', 'Unknown', 'Other']
    
    # Get counts for each category
    counts = categorized.value_counts().reindex(order, fill_value=0)
    
    sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax,
        palette='viridis',
        order=order
    )
    
    ax.set_title(f'Usage Frequency ({usage_col})')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    
    plt.tight_layout()
    return fig

def plot_tool_popularity(df: pd.DataFrame):
    """Bulletproof tool popularity analysis"""
    # Find AI tool column
    possible_cols = [
        col for col in df.columns 
        if any(keyword in col.lower() 
              for keyword in ['tool', 'ai', 'application', 'system', 'tech'])
    ]
    
    if not possible_cols:
        st.error("""
        No AI tool column found. I looked for columns containing:
        'tool', 'ai', 'application', 'system', or 'technology'
        
        Available columns:
        """ + str(df.columns.tolist()))
        return None
    
    tool_col = possible_cols[0]
    
    # Enhanced cleaning
    tool_data = (
        df[tool_col]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({
            'Chatgpt': 'ChatGPT',
            'Gpt': 'ChatGPT',
            'Gpt-3': 'ChatGPT-3',
            'Gpt-4': 'ChatGPT-4',
            'Googlebard': 'Google Bard',
            'Bard': 'Google Bard',
            'Gemini': 'Google Gemini',
            'Githubcopilot': 'GitHub Copilot',
            'Copilot': 'GitHub Copilot',
            'Midjourney': 'Midjourney',
            'Dalle': 'DALL-E',
            'Claude': 'Anthropic Claude'
        }, regex=False)
        .replace(r'^None$|^Nan$|^Null$', 'Unknown', regex=True)
    )
    
    # Count and plot
    tool_counts = tool_data.value_counts().reset_index()
    tool_counts.columns = ['AI Tool', 'Count']
    
    # Limit to top 20 tools if many
    if len(tool_counts) > 20:
        tool_counts = tool_counts.head(20)
    
    fig = px.bar(
        tool_counts,
        x='AI Tool',
        y='Count',
        title=f'AI Tool Popularity ({tool_col})',
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
    st.markdown("""
    This tool analyzes AI tool usage within your organization.
    Upload your data file to get started.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your data (CSV, Excel, or JSON)",
        type=['csv', 'xlsx', 'xls', 'json'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        with st.spinner('Analyzing data...'):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            # Debug information
            with st.expander("🔍 Data Debug Info (click to view)"):
                st.write("### Original Columns:")
                st.write(st.session_state.get('raw_columns', []))
                
                st.write("### Processed Columns:")
                st.write(st.session_state.get('processed_columns', []))
                
                st.write("### Column Mapping:")
                st.write(st.session_state.get('column_mapping', {}))
                
                st.write("### Raw Data Sample:")
                st.write(st.session_state.get('raw_data_sample', {}))
                
                st.write("### Processed Data Sample:")
                st.write(st.session_state.get('processed_data_sample', {}))
            
            st.subheader("📋 Data Overview")
            st.dataframe(df.head(), use_container_width=True)
            
            st.subheader("📈 Usage Analysis")
            col1, col2 = st.columns(2)
            with col1:
                freq_fig = plot_usage_frequency(df)
                if freq_fig:
                    st.pyplot(freq_fig, use_container_width=True)
            
            with col2:
                tool_fig = plot_tool_popularity(df)
                if tool_fig:
                    st.plotly_chart(tool_fig, use_container_width=True)
            
            # Add download button
            st.subheader("💾 Download Processed Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download processed data as CSV",
                data=csv,
                file_name="processed_ai_usage_data.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    create_dashboard()
