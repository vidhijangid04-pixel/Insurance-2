Streamlit Dashboard for Insurance / Policy Status
-----------------------------------------------------

Files:
- app.py : Main Streamlit app (single-file)
- Insurance.csv : Optional sample dataset included (if provided)
- requirements.txt : package names (no pinned versions)

How to deploy:
1. Create a GitHub repo and upload these files at the root (no folders).
2. On Streamlit Cloud, connect the repo and set main file to app.py.
3. Run the app; upload your CSV if not using the bundled sample.

Notes:
- The app uses conservative preprocessing suitable for tree models.
- For production, add hyperparameter tuning, validation and explainability (SHAP) as needed.
