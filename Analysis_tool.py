import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import re
from collections import Counter
from wordcloud import WordCloud

# Updated column mapping to prioritize analysis columns
COLUMN_MAPPING = {
    'analysis_timestamp': ['timestamp', 'date', 'time', 'datetime'],
    'analysis_department': ['department', 'dept', 'division', 'team'],
    'analysis_job_role': ['job_role', 'role', 'position', 'job', 'job role'],
    'analysis_ai_tool': ['ai_tool_used', 'ai tool', 'tool', 'ai', 'ai tool used'],
    'analysis_usage': ['usage_frequency', 'frequency', 'usage', 'how often'],
    'analysis_purpose': ['purpose', 'use case', 'application', 'used for'],
    'analysis_ease': ['ease_of_use', 'ease', 'usability', 'ease of use'],
    'analysis_efficiency': ['time_saved', 'time', 'efficiency', 'time save', 'time saving'],
    'analysis_feedback': ['improvement_suggestion', 'suggestions', 'feedback', 'comments'],
    'analysis_suggestion_category': ['suggestion_category']
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Updated to prioritize analysis columns"""
    # First check if analysis columns already exist
    existing_analysis_cols = [col for col in df.columns if col.startswith('analysis_')]
    if existing_analysis_cols:
        return df
    
    # If not, proceed with standardization
    variation_map = {}
    for std_name, variations in COLUMN_MAPPING.items():
        for var in variations:
            norm_var = var.lower().replace(' ', '_')
            variation_map[norm_var] = std_name
    
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    
    new_columns = []
    seen_columns = set()
    
    for col in df.columns:
        std_col = variation_map.get(col, col)
        
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
        
        df = standardize_columns(df)
        return df.loc[:, ~df.columns.duplicated()]
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_usage_frequency(df: pd.DataFrame):
    """Updated to check both analysis_usage and usage_frequency"""
    usage_col = None
    if 'analysis_usage' in df.columns:
        usage_col = 'analysis_usage'
    elif 'usage_frequency' in df.columns:
        usage_col = 'usage_frequency'
    
    if not usage_col:
        st.warning("No usage frequency data available")
        return None
    
    freq_map = {
        'daily': 'Daily',
        'weekly': 'Weekly',
        'monthly': 'Monthly',
        'rarely': 'Rarely',
        'never': 'Never'
    }
    
    df[usage_col] = df[usage_col].astype(str).str.lower().map(freq_map).fillna('Other')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(
        data=df,
        x=usage_col,
        order=['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never', 'Other'],
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
    """Updated to check both analysis_ai_tool and ai_tool"""
    tool_col = None
    if 'analysis_ai_tool' in df.columns:
        tool_col = 'analysis_ai_tool'
    elif 'ai_tool' in df.columns:
        tool_col = 'ai_tool'
    
    if not tool_col:
        st.warning("No AI tool data available")
        return None
    
    df[tool_col] = df[tool_col].str.strip().str.title()
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
    fig.update_layout(showlegend=False)
    return fig

def plot_metrics(df: pd.DataFrame):
    figs = []
    
    # Check for analysis columns first, then fall back to standard
    ease_col = None
    if 'analysis_ease' in df.columns:
        ease_col = 'analysis_ease'
    else:
        ease_cols = [col for col in df.columns 
                    if ('ease_of_use' in col or 'ease' in col) 
                    and 'timestamp' not in col]
        if ease_cols:
            ease_col = ease_cols[0]
    
    if ease_col:
        fig1 = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y=ease_col, color='skyblue')
        plt.title('Distribution of Ease of Use Ratings')
        plt.ylabel('Rating (1-5)')
        figs.append(fig1)
    
    time_col = None
    if 'analysis_efficiency' in df.columns:
        time_col = 'analysis_efficiency'
    else:
        time_cols = [col for col in df.columns 
                    if ('time_saved' in col or 'time_saving' in col) 
                    and 'timestamp' not in col]
        if time_cols:
            time_col = time_cols[0]
    
    if time_col:
        fig2 = plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, y=time_col, color='lightgreen')
        plt.title('Distribution of Time Saved Ratings')
        plt.ylabel('Rating (1-5)')
        figs.append(fig2)
    
    return figs if figs else None

def analyze_suggestions(df: pd.DataFrame):
    """Updated to check both analysis_feedback and suggestions"""
    feedback_col = None
    category_col = None
    
    if 'analysis_feedback' in df.columns:
        feedback_col = 'analysis_feedback'
    elif 'suggestions' in df.columns:
        feedback_col = 'suggestions'
    
    if 'analysis_suggestion_category' in df.columns:
        category_col = 'analysis_suggestion_category'
    elif 'suggestion_category' in df.columns:
        category_col = 'suggestion_category'
    
    if not feedback_col:
        st.warning("No suggestions data available")
        return
    
    suggestions = df[feedback_col].astype(str).dropna()
    if suggestions.empty:
        st.warning("No suggestions provided in the data")
        return
    
    # Use existing categories if available
    if category_col and category_col in df.columns:
        category_counts = df[category_col].value_counts(normalize=True) * 100
    else:
        # Fall back to categorization logic
        cleaned_suggestions = suggestions.str.lower().str.strip()
        cleaned_suggestions = cleaned_suggestions.replace(
            ['none', 'no', 'n/a', 'nothing', 'no suggestion', 'no improvements', 'nan', 'null'],
            'No suggestions',
            regex=True
        )
        
        categorized = []
        for suggestion in cleaned_suggestions:
            try:
                if suggestion == 'no suggestions':
                    categorized.append('No suggestions')
                elif re.search(r'train|guide|tutorial|documentation|help', suggestion):
                    categorized.append('More training/guidance')
                elif re.search(r'feature|function|capabilit|improve|enhance', suggestion):
                    categorized.append('Feature improvements')
                elif re.search(r'integrat|connect|api|system', suggestion):
                    categorized.append('Better integration')
                elif re.search(r'cost|price|license|subscription', suggestion):
                    categorized.append('Cost reduction')
                elif re.search(r'reliable|accurate|quality|precise|correct', suggestion):
                    categorized.append('Improved accuracy/reliability')
                else:
                    categorized.append('Other suggestions')
            except:
                categorized.append('Other suggestions')
        
        category_counts = pd.Series(categorized).value_counts(normalize=True) * 100
    
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
            text = ' '.join(s for s in suggestions.dropna() if isinstance(s, str))
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
            else:
                st.warning("No valid text for word cloud generation")
        except Exception as e:
            st.warning(f"Could not generate word cloud: {str(e)}")
    
    st.subheader("Most Common Suggestions")
    try:
        common_suggestions = Counter(
            s.capitalize() for s in suggestions.dropna() 
            if isinstance(s, str) and s.lower() not in ['none', 'no', 'n/a', 'nan']
        ).most_common(10)
        
        if common_suggestions:
            st.write("**Top 10 suggestions:**")
            for i, (suggestion, count) in enumerate(common_suggestions, 1):
                st.write(f"{i}. {suggestion} ({count} responses)")
        else:
            st.write("No specific suggestions available")
    except Exception as e:
        st.warning(f"Could not analyze suggestions: {str(e)}")

# Rest of the file remains unchanged...
