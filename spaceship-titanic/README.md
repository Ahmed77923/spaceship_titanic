```

Pipeline/
├── config/              # Configuration Management
│   ├── config.py        # Centralized settings and paths
│   └── __init__.py
├── data/                # Data Processing
│   ├── load_data.py     # Data ingestion and splitting
│   ├── preprocessing.py # Cleaning & preparation pipelines
│   ├── feature_engineering.py # Custom feature creation
│   └── raw/            # Source datasets
├── models/              # ML Modeling
│   ├── base_model.py    # Abstract base classes
│   ├── train_model.py   # Training orchestration
│   ├── evaluate_model.py # Comprehensive evaluation
│   └── hyperparameter_tuning.py # Systematic optimization
├── utils/               # Shared Utilities
│   └── mlflow_utils.py  # Experiment tracking helpers
├── streamlit_app/       # Deployment
│   └── app.py          # Web interface for predictions
└── run_pipeline.py      # Main execution script
```