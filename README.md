# Churn Studio — Group 9

Interactive MBA classroom application for the Netflix-labelled customer churn project.
The dataset's source is unverified; this application is not affiliated with Netflix.

## What you receive

Six sections: Dashboard, Explore Data, Customer Segments, Churn Prediction, Model Comparison and Business Insights.
Original Random Forest and Logistic Regression pipelines include their fitted encoders and scalers. The clustering bundle contains the original fitted scaler, K-Means model, feature order and transformation instructions. Nothing is retrained by the app.

Dashboard data is aggregated from the exported prepared dataset. No customer IDs or customer-level dataset are included in this deployment package. `dashboard_aggregates.csv` replaces a cleaned customer-level CSV for this app's display needs. Models and aggregate data can still disclose information about their training source; get appropriate permission before public publication.

## Download and organise

The easiest route is to download and unzip `netflix_streamlit_deployment.zip`. Open the extracted folder.
If downloading individual files instead, put ALL of them in one folder named `netflix_streamlit`. Put `config.toml` in a subfolder named `.streamlit`.

The main folder must contain:

- app.py
- requirements.txt
- README.md
- random_forest_pipeline.joblib
- logistic_regression_pipeline.joblib
- clustering_bundle.joblib
- dashboard_aggregates.csv
- input_schema.json
- analysis_metadata.json (original Colab environment and limits)
- test_model_comparison.csv
- cross_validation_comparison.csv
- cluster_profiles.csv
- cluster_churn.csv
- cluster_subscription_mix.csv
- random_forest_importance.csv
- logistic_regression_coefficients.csv
- categorical_reference_groups.csv
- validation_report.json
- .streamlit/config.toml

Do not upload the ZIP itself as a substitute for its extracted files. The app expects exact names, including underscores and extensions. Never load joblib files from unknown sources.

## Run locally

Use Python 3.12 (the tested application environment). Install Python from https://www.python.org/downloads/ if needed. Open a terminal in the extracted folder.

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

macOS / Linux / Chromebook with Linux enabled:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the Local URL printed by Streamlit, usually http://localhost:8501. Keep the terminal open. Ctrl+C stops the app. Do not upload `.venv` to GitHub.

## Upload to GitHub through your browser

1. Sign in at https://github.com/ and click New repository.
2. Give it a name such as `netflix-churn-group-9`. Use a private repository until permission to publish project assets is confirmed. Streamlit Community Cloud can use authorised private repositories, subject to your account's limits.
3. Create the repository, then choose Add file > Upload files (or the initial upload link).
4. Drag in all extracted files, keeping app.py and requirements.txt in the repository root, and click Commit changes.
5. Hidden `.streamlit` folders may not appear in your file picker. If needed choose Add file > Create new file, type `.streamlit/config.toml` as its name, paste the supplied config.toml contents, then commit.
6. Check that the three .joblib files are real binary uploads, not renamed text files or links. Each is small enough for ordinary GitHub upload.

## Deploy through Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Allow access to the repository you created. For private repos, grant the necessary GitHub access and check applicable account limits.
3. Click Create app, then choose the option to deploy from an existing repository.
4. Select your repository and branch (usually `main`).
5. Set Main file path to `app.py`.
6. Open Advanced settings and select Python 3.12.
7. Click Deploy. Community Cloud installs requirements.txt automatically; no separate pip command is needed there.
8. Read the build logs if installation fails. Once it opens, perform the checks below before sharing its URL.

This package has been tested locally. It has NOT been deployed or verified on Community Cloud. A deployed URL is only confirmed after the live app opens and its interactions have been checked.

## Verify your live application

- Dashboard, Training: 4,000 records and 50.3% recorded churn.
- Dashboard, Full sample: 5,000 records and 50.3% recorded churn.
- Filter to a plan and region. Counts and plots should change together.
- Explore Data: switch among payment method, gender, device, genre and profile count. Download a group summary.
- Segments: seven groups; cluster 4 has 576 records and 99.65% recorded churn. Changing a segment updates its details.
- Prediction: enter a hypothetical profile and submit. Confirm a score between 0% and 100%, a saved-model class label and a segment number 0–6. Download the hypothetical result.
- Model Comparison: RF accuracy 96.6%, F1 0.9661; LR accuracy 88.2%, F1 0.8839. RF errors: 15 false positives and 19 false negatives.
- Open Business Insights and verify the pilot recommendations and limitations are visible.

## Troubleshooting

- `ModuleNotFoundError`: install using the full requirements.txt command in the same environment. On Cloud check requirements.txt is in the repo root and committed.
- File not found / app cannot load assets: ensure every required file was uploaded with its exact filename and lives beside app.py. Do not leave assets inside an extra folder.
- Version warning or model loading error: use the supplied pins and Python 3.12. Do not install the newest scikit-learn independently; the saved objects were trained with 1.6.1.
- Python 3.12 not available locally: install it, or use Community Cloud with Python 3.12 selected in Advanced settings.
- Browser shows old app: confirm you edited the correct repository and branch; reboot the app from its management controls if necessary.
- App sleeps: reopen it and allow time to wake up.
- No permission for private repo: check GitHub access and the Cloud account's repository visibility settings.
- Do not edit test scores or retrain models to solve a deployment issue. Share the error log instead.

## Methods and responsible interpretation

The fixed stratified train/test split is 80/20, random state 42. Preprocessing is inside each predictive pipeline. Standardised numerical predictors use raw watch_hours; only clustering uses log1p(watch_hours), followed by the original fitted StandardScaler. Cluster fee input is derived from Basic 8.99, Standard 13.99 or Premium 17.99; currency is unspecified.

Seven clusters are descriptive, with overlapping groups and full-training silhouette 0.2255. The K=2 solution was mainly a plan/fee split; seven groups offered richer descriptions with only a modest silhouette increase. Cluster IDs are not risk ranks. Cluster reporting describes training observations only.

RF was selected on three-fold training CV F1 before held-out test evaluation. The final test confusion matrix is [[482, 15], [19, 484]]. Saved model files are copied byte-for-byte from the user's supplied export. The original Colab environment is recorded in analysis_metadata.json. Deployment uses tested Python 3.12 dependencies; NumPy, SciPy and joblib pins differ from that original export. Loading, inference and metric reproduction were tested without changing the models.

The source, event definition, outcome period and pre-churn feature timing are unverified. Scores estimate the recorded label and are not calibrated probabilities of future churn. No threshold optimisation, probability calibration, causal retention benefit, monetary ROI or production readiness is claimed. Numerical LR coefficients are per training standard deviation; categorical odds ratios use the exported reference categories. RF importance is split-based, not causal influence.

Dashboard filters do not change trained models or historical model-comparison scores. All chart summaries are descriptive. Pilot recommendations are proposals, not demonstrated interventions.

## Official deployment references

- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
