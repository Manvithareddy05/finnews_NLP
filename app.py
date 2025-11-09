import streamlit as st
import pandas as pd
import plotly.express as px
import re
import string
import numpy as np
import torch

# --- Import Core NLP Libraries ---
# IMPORTANT: These must be installed in your Conda environment!
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from scipy.special import softmax
    import spacy
    
    # Load FinBERT components and spaCy model globally (cached)
    @st.cache_resource
    def load_nlp_resources():
        """Loads and caches the heavy NLP models (FinBERT and spaCy)."""
        MODEL = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.eval() 
        LABELS = ['Positive', 'Negative', 'Neutral']
        
        # Load the general English spaCy model for NER
        try:
            nlp_ner = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback/install instruction if model isn't downloaded
            st.error("SpaCy model 'en_core_web_sm' not found. Please run: !python -m spacy download en_core_web_sm")
            st.stop()
            
        return tokenizer, model, LABELS, nlp_ner
        
    TOKENIZER, FINBERT_MODEL, LABELS, NLP_NER = load_nlp_resources()
    
except ImportError as e:
    st.error(f"A required library is missing: {e}. Please install it.")
    st.stop()


# --- NLP Helper Functions (Must be defined in app.py) ---

def clean_text(text):
    """Basic text cleaning for model input."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    custom_punc = string.punctuation.replace('$', '').replace('%', '')
    text = text.translate(str.maketrans('', '', custom_punc))
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_finbert_sentiment(text):
    """Runs inference on the FinBERT model."""
    if not text.strip():
        return 'Neutral', 0.0
        
    inputs = TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = FINBERT_MODEL(**inputs)
        
    logits = outputs.logits.numpy()
    scores = softmax(logits, axis=1)[0]
    
    sentiment_index = np.argmax(scores)
    sentiment_label = LABELS[sentiment_index]
    sentiment_score = scores[sentiment_index]
    
    return sentiment_label, sentiment_score

def extract_primary_entity(text):
    """Extracts the most relevant entity (ORG, GPE, PRODUCT) using spaCy."""
    if not text.strip():
        return "Global Market"
        
    doc = NLP_NER(text)
    priority = ["ORG", "GPE", "PRODUCT"] 

    for ent in doc.ents:
        if ent.label_ in priority:
            return ent.text.title()
            
    return "Global Market"

@st.cache_data(show_spinner=True)
def run_full_nlp_pipeline(input_df):
    """Executes the full pipeline on the input DataFrame."""
    
    # 1. Preprocessing
    input_df['Description_Cleaned'] = input_df['Description'].apply(clean_text)

    # 2. FinBERT Sentiment
    # This applies the FinBERT function to the cleaned column
    input_df[['Predicted_Label', 'Confidence_Score']] = input_df['Description_Cleaned'].apply(
        lambda x: pd.Series(get_finbert_sentiment(x))
    )
    
    # 3. Entity Extraction
    input_df['Primary_Entity'] = input_df['Description_Cleaned'].apply(extract_primary_entity)

    # 4. Final Cleanup
    sentiment_order = ['Negative', 'Neutral', 'Positive']
    input_df['Predicted_Label'] = pd.Categorical(
        input_df['Predicted_Label'], categories=sentiment_order, ordered=True
    )
    input_df['Primary_Entity'] = input_df['Primary_Entity'].fillna('Global Market')
    
    return input_df


# --- DASHBOARD DISPLAY FUNCTION (The core visualization logic) ---

def display_dashboard(df_filtered):
    """Generates and displays all KPIs, charts, and tables."""
    
    # --- 1. Key Performance Indicators (KPIs) ---
    total_articles = len(df_filtered)
    positive_count = len(df_filtered[df_filtered['Predicted_Label'] == 'Positive'])
    negative_count = len(df_filtered[df_filtered['Predicted_Label'] == 'Negative'])
    avg_confidence = df_filtered['Confidence_Score'].mean() if not df_filtered.empty else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Articles Analyzed (Filtered)", value=f"{total_articles}")
    with col2:
        st.metric(label="Total Positive News", value=f"{positive_count}")
    with col3:
        st.metric(label="Total Negative News", value=f"{negative_count}")
    with col4:
        st.metric(label="Average Confidence", value=f"{avg_confidence:.2f}")

    st.markdown("---")

    # --- 2. Sentiment Distribution Chart (Pie Chart) ---
    st.header("Overall Sentiment Distribution")
    sentiment_counts = df_filtered['Predicted_Label'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']

    fig_pie = px.pie(
        sentiment_counts, names='Sentiment', values='Count',
        title='Distribution of Predicted Sentiment Labels',
        color='Sentiment',
        color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'orange'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("---")

    # --- 3. Top Negative Entities (Bar Chart) ---
    st.header("Top Entities Driving Negative Sentiment")

    if not df_filtered.empty and len(df_filtered['Primary_Entity'].unique()) > 1:
        entity_summary = df_filtered.groupby('Primary_Entity').agg(
            total_count=('Description', 'count'),
            negative_count=('Predicted_Label', lambda x: (x == 'Negative').sum())
        ).reset_index()
        
        entity_summary['Negative_Ratio'] = entity_summary['negative_count'] / entity_summary['total_count']
        entity_summary = entity_summary[entity_summary['total_count'] >= 5]
        
        top_negative_entities = entity_summary.sort_values(
            by='Negative_Ratio', ascending=False
        ).head(10)

        fig_bar = px.bar(
            top_negative_entities, x='Primary_Entity', y='Negative_Ratio',
            title='Entities with Highest Proportion of Negative News (Min 5 Articles)',
            color='Negative_Ratio', color_continuous_scale=px.colors.sequential.Reds,
            labels={'Negative_Ratio': 'Negative Ratio'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Insufficient data or entities selected to generate the Top Entities chart.")

    st.markdown("---")

    # --- 4. Detail Table View ---
    st.header("Raw Filtered News Data")
    st.dataframe(
        df_filtered[[
            'Primary_Entity', 
            'Predicted_Label', 
            'Confidence_Score', 
            'Description'
        ]].sort_values(by='Confidence_Score', ascending=False),
        use_container_width=True,
        column_config={
            "Description": st.column_config.TextColumn("News Headline", width="large")
        }
    )

# --- Main Application Logic (Runs top-down) ---

st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("Financial News Sentiment Analysis Tool")

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader(
    "Upload Your News CSV",
    type=['csv'],
    help="The CSV must contain a text column named 'Description' with news headlines."
)

if uploaded_file is not None:
    try:
        # Load the uploaded file
        raw_df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    # Validation
    if 'Description' not in raw_df.columns:
        st.error("Error: The uploaded CSV must contain a column named 'Description' for analysis.")
        st.stop()

    st.subheader("Uploaded Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    # --- Trigger Button ---
    run_analysis = st.button("▶️ Run Full NLP Pipeline (Sentiment & Entity Extraction)")
    st.markdown("---")

    if run_analysis:
        # 1. Run the NLP Pipeline
        with st.spinner('Analyzing data with FinBERT and extracting entities... This may take a few moments.'):
            # Use raw_df.copy() to ensure the original DataFrame is not modified by pandas/streamlit
            df_analyzed = run_full_nlp_pipeline(raw_df.copy()) 

        st.success("Analysis Complete! Data is ready for visualization.")
        
        # 2. Display the Filters and Dashboard based on the new data
        
        # --- Sidebar Filters for ANALYZED DATA ---
        st.sidebar.header("Filter Analyzed Data")
        
        # Reset filter options based on the analyzed data
        entity_options = df_analyzed['Primary_Entity'].unique()
        selected_entities = st.sidebar.multiselect(
            "Filter by Entity/Company", options=entity_options, default=entity_options[:5]
        )
        
        sentiment_options = df_analyzed['Predicted_Label'].cat.categories.tolist()
        selected_sentiments = st.sidebar.multiselect(
            "Filter by Sentiment", options=sentiment_options, default=sentiment_options
        )

        min_confidence = st.sidebar.slider(
            "Min Confidence Score", min_value=0.5, max_value=1.0, value=0.75, step=0.01
        )
        
        # --- Apply Filters to the Analyzed Data ---
        df_filtered = df_analyzed[
            (df_analyzed['Primary_Entity'].isin(selected_entities)) &
            (df_analyzed['Predicted_Label'].isin(selected_sentiments)) &
            (df_analyzed['Confidence_Score'] >= min_confidence)
        ]
        
        # 3. Display the full dashboard
        display_dashboard(df_filtered)

else:
    st.info("Upload a CSV file and click 'Run Full NLP Pipeline' to generate the interactive dashboard.")