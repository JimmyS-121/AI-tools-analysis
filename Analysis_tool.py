import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import re
from collections import Counter
from wordcloud import WordCloud

# Enhanced column mapping with more variations
COLUMN_MAPPING = {
    'analysis_timestamp': ['timestamp', 'date', 'time', 'datetime', 'time stamp'],
    'analysis_department': ['department', 'dept', 'division', 'team', 'group'],
    'analysis_job_role': ['job_role', 'role', 'position', 'job', 'job role', 'title'],
    'analysis_ai_tool': ['ai_tool_used', 'ai tool', 'tool', 'ai', 'ai tool used', 'which ai', 'ai system'],
    'analysis_usage': ['usage_frequency', 'frequency', 'usage', 'how often', 'usage frequency', 'how frequently'],
    'analysis_purpose': ['purpose', 'use case', 'application', 'used for', 'primary use'],
    'analysis_ease': ['ease_of_use', 'ease', 'usability', 'ease of use', 'user friendly'],
    'analysis_efficiency': ['time_saved', 'time', 'efficiency', 'time save', 'time saving', 'productivity'],
    'analysis_feedback': ['improvement_suggestion', 'suggestions', 'feedback', 'comments', 'recommendations'],
    'analysis_suggestion_category': ['suggestion_category', 'feedback type', 'category']
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """More robust column standardization"""
    # Create mapping from all variations to standard names
    variation_map = {}
    for std_name, variations in COLUMN_MAPPING.items():
        for var in variations:
            norm_var = var.lower().strip().replace(' ', '_')
            variation_map[norm_var] = std_name
    
    # Normalize all input columns
    norm_columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
    
    # Map to standard names
    new_columns = []
    seen_columns = set()
    
    for i, norm_col in enumerate(norm_columns):
        std_col = variation_map.get(norm_col, df.columns[i])
        
        # Handle duplicate columns
        if std_col in seen_columns:
            counter = 2
            new_col = f"{std_col}_{counter}"
            while new_col in seen_columns:
                counter += 1
                new_col = f"{std_col}_{counter}"
            new_columns.append(new_col)
            seen_columns.add(new_col)
        else:
            new_columns.append(std_col)
            seen_columns.add(std_col)
    
    df.columns = new_columns
    return df.loc[:, ~df.columns.duplicated()]

def load_data(uploaded_file) -> pd.DataFrame:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or JSON file.")
            return None
        
        st.session_state['original_columns'] = df.columns.tolist()
        df = standardize_columns(df)
        st.session_state['standardized_columns'] = df.columns.tolist()
        return df.loc[:, ~df.columns.duplicated()]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_usage_frequency(df: pd.DataFrame):
    """Enhanced usage frequency analysis"""
    usage_col = next((col for col in ['analysis_usage', 'usage_frequency'] if col in df.columns), None)
    
    if not usage_col:
        st.warning("No usage frequency data available")
        st.write("Available columns:", df.columns.tolist())
        return None
    
    # Enhanced frequency mapping
    freq_map = {
        r'daily|every day|each day': 'Daily',
        r'weekly|every week|each week': 'Weekly',
        r'monthly|every month|each month': 'Monthly',
        r'rarely|sometimes|occasionally|seldom': 'Rarely',
        r'never|not at all|haven\'t used': 'Never',
        r'often|regularly|frequently': 'Often'
    }
    
    # Clean and categorize
    usage_series = df[usage_col].astype(str).str.lower().str.strip()
    categorized = usage_series.replace(
        to_replace=list(freq_map.keys()),
        value=list(freq_map.values()),
        regex=True
    ).fillna('Other')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ['Daily', 'Often', 'Weekly', 'Monthly', 'Rarely', 'Never', 'Other']
    sns.countplot(
        x=categorized,
        order=order,
        ax=ax,
        palette='viridis'
    )
    ax.set_title('AI Tools Usage Frequency Distribution')
    ax.set_xlabel('Usage Frequency')
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_tool_popularity(df: pd.DataFrame):
    """Enhanced tool popularity analysis"""
    tool_col = next((col for col in ['analysis_ai_tool', 'ai_tool'] if col in df.columns), None)
    
    if not tool_col:
        st.warning("No AI tool data available")
        st.write("Available columns:", df.columns.tolist())
        return None
    
    # Clean tool names
    df[tool_col] = (
        df[tool_col]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({
            'Chatgpt': 'ChatGPT',
            'Gpt-4': 'ChatGPT-4',
            'Googlebard': 'Google Bard',
            'Githubcopilot': 'GitHub Copilot'
        }, regex=False)
    )
    
    # Count and plot
    tool_counts = df[tool_col].value_counts().reset_index()
    tool_counts.columns = ['AI Tool', 'Count']
    
    fig = px.bar(
        tool_counts,
        x='AI Tool',
        y='Count',
        title='Popularity of AI Tools Among Staff',
        color='AI Tool',
        text='Count'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    return fig

def plot_metrics(df: pd.DataFrame):
    """Enhanced metrics visualization"""
    figs = []
    
    # Ease of use
    ease_col = next((col for col in ['analysis_ease', 'ease_of_use'] if col in df.columns), None)
    if ease_col:
        fig1 = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y=ease_col, color='skyblue')
        plt.title('Distribution of Ease of Use Ratings')
        plt.ylabel('Rating (1-5)')
        figs.append(fig1)
    
    # Time saved
    time_col = next((col for col in ['analysis_efficiency', 'time_saved'] if col in df.columns), None)
    if time_col:
        fig2 = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y=time_col, color='lightgreen')
        plt.title('Distribution of Time Saved Ratings')
        plt.ylabel('Rating (1-5)')
        figs.append(fig2)
    
    return figs if figs else None

def analyze_suggestions(df: pd.DataFrame):
    """Enhanced suggestion analysis"""
    feedback_col = next((col for col in ['analysis_feedback', 'suggestions'] if col in df.columns), None)
    category_col = next((col for col in ['analysis_suggestion_category', 'suggestion_category'] if col in df.columns), None)
    
    if not feedback_col:
        st.warning("No suggestions data available")
        st.write("Available columns:", df.columns.tolist())
        return
    
    suggestions = df[feedback_col].astype(str).dropna()
    if suggestions.empty:
        st.warning("No suggestions provided in the data")
        return
    
    # Use existing categories if available
    if category_col and category_col in df.columns:
        category_counts = df[category_col].value_counts(normalize=True) * 100
    else:
        # Enhanced categorization
        patterns = {
            'Training/guidance': r'train|guide|tutorial|doc|manual|help|learn|educate',
            'Feature improvements': r'feature|function|tool|option|capability|improve|enhance|add',
            'Better integration': r'integrat|connect|api|system|plugin|bridge|import|export',
            'Cost reduction': r'cost|price|cheap|afford|license|subscription|fee|pay',
            'Improved accuracy': r'reliable|accurate|quality|precise|correct|better|trust|depend',
            'No suggestions': r'no|none|n/a|not|nothing|nil|nan|null|undefined'
        }
        
        categorized = []
        for suggestion in suggestions.str.lower():
            matched = False
            for category, pattern in patterns.items():
                if re.search(pattern, suggestion):
                    categorized.append(category)
                    matched = True
                    break
            if not matched:
                categorized.append('Other suggestions')
        
        category_counts = pd.Series(categorized).value_counts(normalize=True) * 100
    
    # Visualization
    st.subheader("Suggestions Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        if not category_counts.empty:
            fig1 = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Suggestion Categories Distribution',
                hole=0.3
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        try:
            text = ' '.join(s for s in suggestions if isinstance(s, str))
            if text.strip():
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(text)
                fig2 = plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Common Words in Suggestions')
                st.pyplot(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")
    
    # Top suggestions
    st.subheader("Most Common Suggestions")
    try:
        common_suggestions = Counter(
            s.strip().capitalize() for s in suggestions 
            if isinstance(s, str) and len(s.strip()) > 3
        ).most_common(10)
        
        if common_suggestions:
            for i, (suggestion, count) in enumerate(common_suggestions, 1):
                st.write(f"{i}. {suggestion} ({count} responses)")
        else:
            st.write("No specific suggestions available")
    except Exception as e:
        st.warning(f"Could not analyze suggestions: {str(e)}")

def create_dashboard():
    st.set_page_config(
        page_title="AI Tools Usage Analysis",
        layout="wide",
        page_icon="🤖"
    )
    
    st.title("📊 AI Tools Usage Analysis Dashboard")
    st.markdown("""
    This tool helps analyze how AI tools are being used within your organization.
    Upload your data file to get started.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload your data (CSV or JSON)",
        type=['csv', 'json'],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        with st.spinner('Analyzing data...'):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            # Debug information (can be collapsed)
            with st.expander("🔍 Data Debug Info (click to view)"):
                st.write("Original columns:", st.session_state.get('original_columns', []))
                st.write("Standardized columns:", st.session_state.get('standardized_columns', []))
                st.write("Sample data:", df.head(3))
            
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
            
            st.subheader("⭐ User Experience Metrics")
            metric_figs = plot_metrics(df)
            if metric_figs:
                cols = st.columns(len(metric_figs))
                for i, fig in enumerate(metric_figs):
                    with cols[i]:
                        st.pyplot(fig, use_container_width=True)
            
            st.subheader("💡 User Feedback Analysis")
            analyze_suggestions(df)
            
            # Add download button
            st.subheader("💾 Download Analyzed Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download analyzed data as CSV",
                data=csv,
                file_name="ai_usage_analysis.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    create_dashboard()
