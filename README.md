# hfrs_hybrid_pipeline

Hybrid HFRS pipeline reusing logic from the original clinical_fragility pipeline.

## Streamlit App Deployment & Usage

### Local Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

### Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and sign in with your GitHub account.
3. Click "New app", select your repo, and set the app path to `hfrs_hybrid_pipeline/app.py`.
4. Click "Deploy".

### App Features
- Paste or upload clinical notes for frailty scoring and diagnostics extraction.
- Outputs fragility score, category, diagnostics, and Z-code descriptions.
- Download results as CSV.

### Notes
- Make sure all required data files and models are available or documented for deployment.
- For large models or private data, consider self-hosting.
