This repository contains an automated pipeline for air quality prediction using real-time environmental data. It ingests data, performs feature engineering, trains machine learning models, and registers them in Hopsworks. The pipeline is orchestrated using GitHub Actions and optionally supports a Streamlit front end for manual execution and visualization.

#Features

- Hourly automation via GitHub Actions
- Feature engineering from AQI and weather data
- Model training using scikit-learn and XGBoost
- Versioned model registry in Hopsworks
- Secure secrets management using GitHub Secrets
- Optional Streamlit interface for manual runs

# Project Structure

.
├── AQIAgent.py              # Main pipeline entry point
├── .github/
│   └── workflows/
│       └── aqi_pipeline.yml # GitHub Actions workflow
├── models/                  # Trained model artifacts
└── README.md
└── front.py

# Automation

The pipeline runs automatically every hour using GitHub Actions. The workflow is defined in `.github/workflows/aqi_pipeline.yml` and is triggered by a cron schedule:

yaml
on:
  schedule:
    - cron: "0 * * * *"

# Secrets

Sensitive credentials are stored securely using GitHub Secrets. 

# Manual Execution

To run the pipeline manually via Streamlit:

streamlit run AQIAgent.py frontend

This launches a user interface for triggering feature extraction and model training.

# Outputs

- Trained models saved in `models/`
- Registered in Hopsworks with versioning
- Logs available in GitHub Actions UI

