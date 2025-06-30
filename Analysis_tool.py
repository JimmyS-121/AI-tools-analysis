import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

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
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None

        if df.empty:
            st.error("Uploaded file is empty")
            return None

        # Validate columns
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None

        return df

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def plot_usage_frequency(df):
    """Visualize usage frequency data"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Order categories logically
    order = ['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never']
    
    # Get counts and ensure all categories are represented
    counts = df['usage_frequency'].value_counts().reindex(order, fill_value=0)
    
    # Create plot
    sns.barplot(
        x=counts.index,
        y=counts.values,
        order=order,
        palette='viridis',
        ax=ax
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
        
