import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
from collections import Counter

# Expected column names (must match data cleaner output)
EXPECTED_COLUMNS = {
    'timestamp': 'datetime64[ns]',
    'department': 'object',
    'job_role': 'object',
    'ai_tool': 'object',
    'usage_frequency': 'object',
    'purpose': 'object',
    'ease_of_use': 'float64',
    'efficiency': 'float64',
    'suggestions': 'object'
}

def load_data(uploaded_file):
    """Load and validate data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(uploaded_file)
            except ImportError:
                st.error("Excel support requires openpyxl. Please install with: pip install openpyxl")
                return None
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        
        # Validate columns and data types
        for col, dtype in EXPECTED_COLUMNS.items():
            if col not in df.columns:
                st.error(f"Missing expected column: {col}")
                return None
            if not pd.api.types.is_dtype_equal(df[col].dtype, dtype):
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    st.error(f"Could not convert column '{col}' to expected type {dtype}")
                    return None
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def plot_usage_frequency(df):
    """Visualize usage frequency data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define expected categories and ensure all are represented
    categories = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
    
    # Get counts and ensure all categories are represented
    counts = df['usage_frequency'].value_counts().reindex(categories, fill_value=0)
    
    # Create plot with hue parameter to address warning
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,  # Added to address warning
        palette='viridis',
        ax=ax,
        legend=False  # Added to address warning
    )
    
    # Customize plot
    ax.set_title('AI Tools Usage Frequency', pad=20)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Number of Respondents')
    plt.xticks(rotation=45)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 5),
            textcoords='offset points'
        )
    
    plt.tight_layout()
    return fig

def plot_tool_popularity(df):
    """Visualize AI tool popularity"""
    # Get tool counts
    tool_counts = df['ai_tool'].value_counts().reset_index()
    tool_counts.columns = ['AI Tool', 'Count']
    
    # Create interactive plot
    fig = px.bar(
        tool_counts,
        x='AI Tool',
        y='Count',
        title='AI Tool Popularity',
        color='AI Tool',
        text='Count'
    )
    
    # Customize plot
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        xaxis_tickangle=-45,
        height=600,
        margin=dict(t=60)
    )
    
    return fig

def plot_department_usage(df):
    """Visualize usage by department"""
    try:
        if 'department' not in df.columns:
            return None
            
        # Define expected categories
        categories = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
        
        # Create cross-tab of department vs usage frequency
        cross_tab = pd.crosstab(
            index=df['department'],
            columns=df['usage_frequency'],
            normalize='index'
        ).fillna(0)
        
        # Ensure all categories are present (add missing ones with 0 values)
        for cat in categories:
            if cat not in cross_tab.columns:
                cross_tab[cat] = 0
        
        # Only reorder with columns that exist
        existing_cols = [col for col in categories if col in cross_tab.columns]
        cross_tab = cross_tab[existing_cols]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        cross_tab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        # Customize plot
        ax.set_title('AI Usage Frequency by Department', pad=20)
        ax.set_xlabel('Department')
        ax.set_ylabel('Percentage of Respondents')
        ax.legend(title='Usage Frequency', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating department usage plot: {str(e)}")
        return None

def plot_suggestions_wordcloud(df):
    """Generate word cloud from suggestions"""
    try:
        if 'suggestions' not in df.columns:
            return None
            
        # Combine all suggestions
        text = ' '.join(df['suggestions'].dropna().astype(str))
        
        if not text.strip():
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50
        ).generate(text)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Common Words in Suggestions', pad=20)
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

def show_data_summary(df):
    """Display key data metrics"""
    st.subheader("📊 Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Responses", len(df))
    
    with col2:
        st.metric("Unique AI Tools", df['ai_tool'].nunique())
    
    with col3:
        avg_ease = df['ease_of_use'].mean()
        st.metric("Average Ease of Use", f"{avg_ease:.1f}/5")

def create_dashboard():
    """Main dashboard function"""
    st.set_page_config(
        page_title="AI Tools Usage Dashboard",
        layout="wide",
        page_icon="🤖"
    )
    
    st.title("AI Tools Usage Analysis Dashboard")
    st.markdown("Analyze how AI tools are being used across your organization.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your cleaned data file (CSV or Excel)",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        with st.spinner('Processing data...'):
            df = load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            # Data preview
            with st.expander("🔍 View Raw Data"):
                st.dataframe(df)
            
            # Main analysis section
            show_data_summary(df)
            
            st.subheader("Usage Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.pyplot(plot_usage_frequency(df))
            
            with col2:
                st.plotly_chart(plot_tool_popularity(df), use_container_width=True)
            
            # Department breakdown
            dept_fig = plot_department_usage(df)
            if dept_fig:
                st.subheader("Department-wise Usage Patterns")
                st.pyplot(dept_fig)
            
            # Suggestions analysis
            st.subheader("User Suggestions Analysis")
            suggestions_fig = plot_suggestions_wordcloud(df)
            if suggestions_fig:
                st.pyplot(suggestions_fig)
            else:
                st.info("No suggestions data available for analysis")
            
            # Data download
            st.subheader("Download Processed Data")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="ai_usage_analysis.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    create_dashboard()
