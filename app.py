import warnings
warnings.filterwarnings("ignore")
import os
import streamlit as st
import pandas as pd
from hfrs_hybrid_pipeline.processing import ClinicalNoteProcessor
from medkit.text.ner.quick_umls_matcher import QuickUMLSMatcher


# --- CONFIG ---
CUI_TO_CIM10_PATH = '/home/coder/nalfe/data/cui_to_cim10.json'
HFRS_POINTS_PATH = '/home/coder/nalfe/data/hfrs_points.csv'
QUICKUMLS_PATH = '/home/coder/nalfe/quickumls_installs/2025AA_fre_lower'

# Register QuickUMLS install before using ClinicalNoteProcessor
QuickUMLSMatcher.add_install(
    path=QUICKUMLS_PATH,
    language="FRE",
    version="2025AA",
    lowercase=True,
    normalize_unicode=False,
)

# Set this to your Ollama model name or a supported LLM pipeline
LLM_MODEL_NAME = "ollama:mistral:latest"  # or another supported model

@st.cache_resource
def get_processor():
    # You can pass additional config as needed
    return ClinicalNoteProcessor({}, llm_model_name=LLM_MODEL_NAME)

processor = get_processor()

st.title('Early Frailty Assessment from French Clinical Notes Using NLP in Hospitalized Older Adults')

input_mode = st.radio('Input mode', ['Paste text', 'Upload text file'])
texts = []
if input_mode == 'Paste text':
    note_text = st.text_area('Paste clinical note:', height=200)
    if note_text.strip():
        texts = [note_text.strip()]
else:
    uploaded_file = st.file_uploader('Upload a .txt, .csv, or .xlsx file (one note per block, separated by --- for .txt)', type=['txt', 'csv', 'xlsx'])
    if uploaded_file:
        if uploaded_file.name.endswith('.txt'):
            file_content = uploaded_file.read().decode('utf-8')
            texts = [t.strip() for t in file_content.split('---') if t.strip()]
        elif uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
            # Try to find a column with note text
            text_col = None
            for col in df_input.columns:
                if 'text' in col.lower():
                    text_col = col
                    break
            if text_col is None:
                st.error('No text column found in CSV. Please ensure a column contains note text.')
                texts = []
            else:
                texts = df_input[text_col].astype(str).tolist()
        elif uploaded_file.name.endswith('.xlsx'):
            df_input = pd.read_excel(uploaded_file)
            text_col = None
            for col in df_input.columns:
                if 'text' in col.lower():
                    text_col = col
                    break
            if text_col is None:
                st.error('No text column found in Excel file. Please ensure a column contains note text.')
                texts = []
            else:
                texts = df_input[text_col].astype(str).tolist()

if st.button('Analyze') and texts:
    with st.spinner('Processing...'):
        df = processor.process_notes(pd.DataFrame({"text": texts}), text_col="text")
        st.subheader('Results')
        for i, row in df.iterrows():
            st.write(f"**Note {i+1}:**")
            st.write(f"Fragility Score: {round(row['HFRS_score_quickumls'], 1)}")
            category = row['fragility_category'] if 'fragility_category' in row and pd.notna(row['fragility_category']) else 'unknown'
            st.write(f"Category: {category}")
            st.write('Diagnostics and Codes:')
            st.dataframe(pd.DataFrame(row['diagnostics_cim10_codes_quickumls']))
        csv = df.to_csv(index=False).encode('utf-8')
        #st.download_button('Download results as CSV', csv, 'hfrs_hybrid_results.csv', 'text/csv')
else:
    st.info('Paste or upload clinical notes and click Analyze.')
