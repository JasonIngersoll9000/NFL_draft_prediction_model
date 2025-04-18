# NFL Quarterback Draft Analysis

A machine learning approach to predicting quarterback success and evaluating NFL draft efficiency.

## Project Overview

This repository contains the code, models, and results for analyzing NFL quarterback draft outcomes. The project uses college statistics, physical measurements, and draft information to predict both the likelihood of being drafted and expected career performance measured through weighted Approximate Value (wAV).

## Repository Structure

```
quarterback-draft-analysis/
├── data/processed/          # Processed data files (sample data included)
├── models/                  # Trained machine learning models
├── results/                 # Analysis results and visualizations
│   ├── draft_analysis/      # Draft value visualizations
│   └── quarterback/         # QB prediction results
└── src/                     # Source code
    ├── data/                # Data processing scripts
    ├── features/            # Feature engineering code
    ├── models/              # Model training code
    └── visualization/       # Visualization scripts
```

## Key Features

- Predictive models for quarterback draft status and NFL performance
- Draft value analysis showing the relationship between draft position and QB value
- Visualizations of quarterback performance metrics and draft efficiency
- Analysis of factors affecting successful college-to-NFL quarterback transitions

## Data

The repository includes sample data files in `data/processed/`. Due to size limitations, the full player database is not included, but sample files demonstrate the data structure.

## Usage

```bash
# Clone the repository
git clone https://github.com/JasonIngersoll9000/NFL_draft_prediction_model.git
cd NFL_draft_prediction_model

# Install dependencies
pip install -r requirements.txt

# Run visualizations
python src/visualization/visualize_qb_results.py
python src/visualization/visualize_draft_pick_value.py
```

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```

## License

MIT License
