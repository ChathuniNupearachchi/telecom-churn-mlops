# Telecom Customer Churn Prediction - MLOps Pipeline

An end-to-end machine learning operations project demonstrating production-grade ML deployment

## Project Status

**Work in Progress** - Currently in development

## Business Problem

Predict customer churn in a telecom company to enable proactive retention strategies and reduce revenue loss.

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Tracking**: MLflow
- **API**: FastAPI
- **Monitoring**: Evidently
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Cloud**: AWS/GCP (planned)

## Project Structure

```
telecom-churn-mlops/
├── data/              # Data storage
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
├── tests/             # Unit tests
├── config/            # Configuration files
├── models/            # Saved models
└── docker/            # Docker configuration
```

## Getting Started

```bash
# Clone repository
git clone <your-repo-url>
cd telecom-churn-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```
